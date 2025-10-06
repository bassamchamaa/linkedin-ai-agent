import os
import json
import re
import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
import xml.etree.ElementTree as ET
import requests


class LinkedInAIAgent:
    def __init__(self):
        # Secrets
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()  # optional fallback

        # Topic rotation
        self.topics = {
            "tech_partnerships": [
                "technology partnerships business development",
                "B2B tech strategic partnerships",
                "enterprise technology alliances",
            ],
            "ai": [
                "artificial intelligence enterprise",
                "AI business applications",
                "machine learning partnerships",
            ],
            "payments": [
                "fintech payments partnerships",
                "payment technology trends",
                "digital payments innovation",
            ],
        }

        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("LinkedIn secrets missing. Check LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key and not self.openai_key:
            print("No model key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")

    # -----------------------------
    # Style helpers
    # -----------------------------
    def enforce_style_rules(self, text):
        # No em or en dashes, no semicolons
        text = text.replace("—", ",").replace("–", ",").replace(";", ",")
        # Kill “As a …” openers
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)
        return text.strip()

    # -----------------------------
    # State helpers
    # -----------------------------
    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"last_topics": [], "post_count": 0, "last_post_date": None}

    def save_state(self, state):
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Could not save state: {e}")

    def should_post_today(self, state):
        today = datetime.now().strftime("%Y-%m-%d")
        if state.get("last_post_date") == today:
            print(f"Already posted today ({today}). Skipping.")
            return False
        return True

    def get_next_topic(self, state):
        last_topics = state.get("last_topics", [])
        available = [t for t in self.topics if t not in last_topics]
        if not available:
            last_topics = []
            available = list(self.topics.keys())

        next_topic = random.choice(available)
        last_topics.append(next_topic)
        if len(last_topics) > 2:
            last_topics = last_topics[-2:]

        state["last_topics"] = last_topics
        return next_topic, state

    # -----------------------------
    # URL resolution
    # -----------------------------
    def _extract_original_from_link(self, link):
        """
        Google News links can be:
          - Aggregator with ?url= param
          - /rss/articles/... that 302s to the publisher
          - Plain news.google.com
        We try query param, then follow redirects, else return original.
        """
        try:
            if not link:
                return ""

            parsed = urlparse(link)
            if "news.google.com" not in parsed.netloc:
                return link  # already a publisher link

            # If there's a url= param, prefer it
            q = parse_qs(parsed.query)
            if "url" in q and q["url"]:
                return unquote(q["url"][0])

            # Try resolving redirects to publisher
            try:
                r = requests.get(link, timeout=10, allow_redirects=True, headers={"User-Agent": "curl/8"})
                final_url = r.url
                if final_url and "news.google.com" not in urlparse(final_url).netloc:
                    return final_url
            except Exception:
                pass

            return link  # fallback
        except Exception:
            return link

    def _extract_publisher_from_description(self, description):
        """
        Some RSS items embed a publisher link inside <description>.
        Grab the first http(s) link that is not news.google.com.
        """
        if not description:
            return ""
        try:
            urls = re.findall(r'href="(https?://[^"]+)"', description)
            for u in urls:
                if "news.google.com" not in urlparse(u).netloc:
                    return u
        except Exception:
            pass
        return ""

    # -----------------------------
    # News fetch via Google News RSS
    # -----------------------------
    def fetch_news(self, topic_key, query, max_items=6):
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items = []
        try:
            r = requests.get(rss, timeout=12, headers={"User-Agent": "curl/8"})
            if r.status_code != 200:
                print(f"News fetch error {r.status_code} for {topic_key}: {r.text[:200]}")
                return items

            root = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None:
                return items

            for it in channel.findall("item")[: max_items * 2]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                desc = (it.findtext("description") or "").strip()

                # Prefer publisher link from description
                publisher_link = self._extract_publisher_from_description(desc)
                best = publisher_link or self._extract_original_from_link(link)

                items.append({"title": title, "link": best})

                if len(items) >= max_items:
                    break

        except Exception as e:
            print(f"Error parsing RSS for {topic_key}: {e}")

        if not items:
            print(f"⚠ No news found for {topic_key}, using fallback.")
            items = [{"title": f"Current industry trends in {topic_key.replace('_', ' ')}", "link": ""}]
        return items

    # -----------------------------
    # Prompt helpers
    # -----------------------------
    def _build_prompt(self, topic_key, news_items, include_link, keep=2, words_low=120, words_high=160):
        trimmed = []
        for it in news_items[:keep]:
            title = it["title"][:110] + ("..." if len(it["title"]) > 110 else "")
            link = it.get("link") or ""
            # Only include specific publisher links in the prompt context
            link_part = f" | {link}" if link and "news.google.com" not in link else ""
            trimmed.append(f"- {title}{link_part}")

        news_context = "\n".join(trimmed)
        link_instruction = (
            "Include exactly one publisher link in the body."
            if include_link and any(x.get("link") and "news.google.com" not in x["link"] for x in news_items)
            else "Do not include any links."
        )

        prompt = (
            f"Write a LinkedIn post from a senior sales leader in tech and fintech who favors partnerships. "
            f"Topic: {topic_key.replace('_', ' ')}.\n\n"
            f"Keep it {words_low} to {words_high} words. Be direct, practical, and human. "
            f"Avoid generic set-ups like 'As a ...'. No em dashes. No semicolons. "
            f"End with two or three relevant hashtags. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )
        return prompt

    # -----------------------------
    # Gemini generation (try known models directly)
    # -----------------------------
    def _extract_text_from_gemini(self, payload):
        try:
            cand = payload["candidates"][0]
        except Exception:
            return None

        content = cand.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                if any(texts):
                    return "\n".join(t for t in texts if t)
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
            if any(texts):
                return "\n".join(t for t in texts if t)
        if "text" in cand:
            return cand["text"]
        return None

    def generate_post_with_gemini(self, topic_key, news_items, include_link):
        if not self.gemini_key:
            return None

        attempts = [
            {"api": "v1beta", "model": "gemini-1.5-flash-latest", "keep": 2, "max_out": 520, "words": (120, 160)},
            {"api": "v1beta", "model": "gemini-1.5-flash",        "keep": 2, "max_out": 480, "words": (110, 150)},
            {"api": "v1beta", "model": "gemini-1.5-pro-latest",   "keep": 1, "max_out": 480, "words": (110, 150)},
            {"api": "v1beta", "model": "gemini-1.5-pro",          "keep": 1, "max_out": 440, "words": (100, 140)},
            {"api": "v1",     "model": "gemini-2.5-flash",        "keep": 1, "max_out": 600, "words": (120, 160)},
        ]

        headers = {"Content-Type": "application/json"}

        for i, step in enumerate(attempts, start=1):
            base = (
                f"https://generativelanguage.googleapis.com/"
                f"{step['api']}/models/{step['model']}:generateContent"
            )
            prompt = self._build_prompt(
                topic_key, news_items, include_link,
                keep=step["keep"], words_low=step["words"][0], words_high=step["words"][1]
            )
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": step["max_out"]},
            }

            print(f"Attempt {i}: {step['model']} on {step['api']} with maxOutputTokens={step['max_out']}")
            try:
                resp = requests.post(f"{base}?key={self.gemini_key}", headers=headers, json=body, timeout=45)
                if resp.status_code == 404:
                    print(f"Model not found on attempt {i}: {step['model']}")
                    continue
                if resp.status_code != 200:
                    print(f"Gemini error {resp.status_code} on attempt {i}: {resp.text[:600]}")
                    continue

                data = resp.json()
                text = self._extract_text_from_gemini(data)
                if not text:
                    fr = None
                    try:
                        fr = data["candidates"][0].get("finishReason")
                    except Exception:
                        pass
                    if fr:
                        print(f"Finish reason: {fr}")
                    print("Gemini response preview:", json.dumps(data)[:600])
                    continue

                text = self.enforce_style_rules(text)

                # Ensure any included link is a real publisher URL
                if include_link:
                    has_link = "http://" in text or "https://" in text
                    if not has_link:
                        good = next((it["link"] for it in news_items
                                     if it.get("link") and "news.google.com" not in it["link"]), "")
                        if good:
                            text += f"\n\n{good}"
                            text = self.enforce_style_rules(text)
                        else:
                            # No clean publisher link, switch to thought piece
                            text = re.sub(r"\nhttps?://\S+$", "", text).strip()

                return text

            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue

        return None

    # -----------------------------
    # OpenAI fallback (optional)
    # -----------------------------
    def generate_post_with_openai(self, topic_key, news_items, include_link):
        if not self.openai_key:
            return None

        news_context = "\n".join(
            f"- {it['title'][:110]}{'...' if len(it['title'])>110 else ''}"
            f"{' | ' + it['link'] if it.get('link') and 'news.google.com' not in it['link'] else ''}"
            for it in news_items[:2]
        )
        link_instruction = (
            "Include exactly one publisher link in the body."
            if include_link and any(x.get("link") and "news.google.com" not in x["link"] for x in news_items)
            else "Do not include any links."
        )
        prompt = (
            f"Write a LinkedIn post from a senior sales leader in tech and fintech who favors partnerships. "
            f"Topic: {topic_key.replace('_', ' ')}.\n\n"
            f"Keep it 120 to 160 words. Be direct, practical, and human. "
            f"Avoid generic set-ups like 'As a ...'. No em dashes. No semicolons. "
            f"End with two or three relevant hashtags. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 450}
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=45)
            if r.status_code != 200:
                print(f"OpenAI error {r.status_code}: {r.text[:400]}")
                return None
            text = r.json()["choices"][0]["message"]["content"]
            text = self.enforce_style_rules(text)

            if include_link and ("http://" not in text and "https://" not in text):
                good = next((it["link"] for it in news_items if it.get("link") and "news.google.com" not in it["link"]), "")
                if good:
                    text += f"\n\n{good}"
                    text = self.enforce_style_rules(text)
            return text
        except Exception as e:
            print(f"OpenAI call failed: {e}")
            return None

    # -----------------------------
    # Last-ditch local generator
    # -----------------------------
    def generate_post_locally(self, topic_key, news_items, include_link):
        bullets = [it["title"] for it in news_items[:2]]
        hook = "Partnerships deliver when the motion is clear, measurable, and repeatable."
        insight = (
            f"Right now I’m seeing faster wins when teams anchor around one customer use case, "
            f"ship a tight integration, co-sell with the partner, and report the revenue impact in the same dashboard."
        )
        close = "Pick a single motion, get proof fast, then scale across your partner ecosystem."
        link_line = ""
        if include_link:
            good = next((it["link"] for it in news_items if it.get("link") and "news.google.com" not in it["link"]), "")
            if good:
                link_line = f"\n\n{good}"
        post = (
            f"{hook} {insight} {close}\n\n"
            f"#partnerships #busdev #fintech{link_line}"
        )
        return self.enforce_style_rules(post)

    # -----------------------------
    # LinkedIn posting
    # -----------------------------
    def post_to_linkedin(self, text):
        if not self.linkedin_token or not self.person_urn:
            print("Missing LinkedIn token or Person URN. Cannot post.")
            return False

        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {self.linkedin_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        payload = {
            "author": f"urn:li:person:{self.person_urn}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            if r.status_code in (200, 201):
                print("✓ Successfully posted to LinkedIn")
                return True
            print(f"✗ Failed to post: {r.status_code} - {r.text[:400]}")
            if r.status_code == 401:
                print("Hint: token expired or missing w_member_social scope.")
            if r.status_code == 403:
                print("Hint: app lacks permission to post as a member.")
            return False
        except Exception as e:
            print(f"✗ Error posting to LinkedIn: {e}")
            return False

    # -----------------------------
    # Main run
    # -----------------------------
    def run_weekly_post(self):
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        state = self.load_state()
        if not self.should_post_today(state):
            return

        topic_key, state = self.get_next_topic(state)
        include_link = random.random() < 0.6
        post_type = "with publisher link" if include_link else "thought leadership piece"

        print(f"--- Topic: {topic_key.replace('_', ' ').title()} ({post_type}) ---")
        query = random.choice(self.topics[topic_key])
        print(f"Fetching news for: {query}")
        news_items = self.fetch_news(topic_key, query)
        print(f"Found {len(news_items)} news items")

        # If we decided to include a link but none of the items has a clean publisher link, flip to thought piece
        if include_link and not any(it.get("link") and "news.google.com" not in it["link"] for it in news_items):
            include_link = False
            print("No clean publisher link found, switching to thought leadership.")

        # Generate with Gemini, else OpenAI, else local
        post_text = None
        if self.gemini_key:
            print("Generating post with Gemini...")
            post_text = self.generate_post_with_gemini(topic_key, news_items, include_link)
        if not post_text and self.openai_key:
            print("Gemini failed, using OpenAI...")
            post_text = self.generate_post_with_openai(topic_key, news_items, include_link)
        if not post_text:
            print("All model calls failed, using local template.")
            post_text = self.generate_post_locally(topic_key, news_items, include_link)

        print("\nGenerated post:\n" + "-" * 60)
        print(post_text)
        print("-" * 60)

        ok = self.post_to_linkedin(post_text)
        if ok:
            state["post_count"] = state.get("post_count", 0) + 1
            state["last_post_date"] = datetime.now().strftime("%Y-%m-%d")
            self.save_state(state)
            print(f"\n✅ Post #{state['post_count']} published successfully!")
            remaining = 3 - len(state.get("last_topics", []))
            print(f"Next topics: {[t for t in self.topics if t not in state['last_topics']]}")
            print(f"Posts remaining in cycle window: {remaining}")
        else:
            print("\n❌ Failed to publish post")


if __name__ == "__main__":
    agent = LinkedInAIAgent()
    agent.run_weekly_post()
