import os
import json
import re
import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
import xml.etree.ElementTree as ET
import requests


# =============== Utility ===============

def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_google_news(url: str) -> bool:
    d = domain(url)
    return d.endswith("news.google.com") or d.endswith("google.com")


# =============== Agent ===============

class LinkedInAIAgent:
    def __init__(self):
        # Secrets
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()  # optional fallback

        # Topics and queries
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

        # Hashtag pools by topic
        self.topic_tags = {
            "tech_partnerships": ["#partnerships", "#busdev", "#GTM", "#SaaS"],
            "ai": ["#AI", "#EnterpriseAI", "#B2B", "#GTM"],
            "payments": ["#fintech", "#payments", "#partnerships", "#B2B"],
        }

        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("LinkedIn secrets missing. Check LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key and not self.openai_key:
            print("No model key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")

    # =============== Style & Quality ===============

    def enforce_style_rules(self, text: str) -> str:
        # No em/en dashes, no semicolons
        text = text.replace("—", ",").replace("–", ",").replace(";", ",")
        # Kill generic "As a ..." openers
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)
        # Remove unmatched starting quote
        if text.count('"') % 2 == 1 and text.strip().startswith('"'):
            text = text.lstrip('"').lstrip()
        # Collapse double spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def debuzz(self, text: str) -> str:
        # Light touch de-buzz
        replacements = {
            r"\bsynergy\b": "fit",
            r"\bleverage\b": "use",
            r"\bparadigm\b": "approach",
            r"\bgame[- ]changer\b": "big step",
            r"\bcutting edge\b": "leading",
            r"\bdisrupt(ive|ion)\b": "shift",
        }
        for pat, rep in replacements.items():
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text

    def ends_cleanly(self, text: str) -> bool:
        return bool(re.search(r'[.!?]"?\s*$', text))

    def has_hashtags(self, text: str) -> bool:
        return len(re.findall(r"(?:^|\s)#\w+", text)) >= 2

    # =============== State ===============

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

    def should_post_today(self, state) -> bool:
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

    # =============== News & Links ===============

    def _extract_original_from_link(self, link: str) -> str:
        try:
            if not link:
                return ""
            parsed = urlparse(link)
            if "news.google.com" not in parsed.netloc:
                return link
            q = parse_qs(parsed.query)
            if "url" in q and q["url"]:
                return unquote(q["url"][0])
            # Follow redirects to publisher
            try:
                r = requests.get(link, timeout=10, allow_redirects=True, headers={"User-Agent": "curl/8"})
                final_url = r.url
                if final_url and not is_google_news(final_url):
                    return final_url
            except Exception:
                pass
            return link
        except Exception:
            return link

    def _extract_publisher_from_description(self, description: str) -> str:
        if not description:
            return ""
        try:
            urls = re.findall(r'href="(https?://[^"]+)"', description)
            for u in urls:
                if not is_google_news(u):
                    return u
        except Exception:
            pass
        return ""

    def fetch_news(self, topic_key: str, query: str, max_items: int = 6):
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

                pub_link = self._extract_publisher_from_description(desc)
                best = pub_link or self._extract_original_from_link(link)
                items.append({"title": title, "link": best})

                if len(items) >= max_items:
                    break

        except Exception as e:
            print(f"Error parsing RSS for {topic_key}: {e}")

        if not items:
            print(f"⚠ No news found for {topic_key}, using fallback.")
            items = [{"title": f"Current industry trends in {topic_key.replace('_', ' ')}", "link": ""}]
        return items

    def pick_publisher_link(self, news_items) -> str:
        for it in news_items:
            l = (it.get("link") or "").strip()
            if l and not is_google_news(l):
                # Filter out pure homepage links if we can
                path = urlparse(l).path or ""
                if path and path != "/":
                    return l
        return ""

    # =============== Prompting ===============

    def _build_prompt(self, topic_key, news_items, include_link, tone, keep=2, words_low=120, words_high=170) -> str:
        trimmed = []
        for it in news_items[:keep]:
            t = it["title"]
            title = t[:110] + ("..." if len(t) > 110 else "")
            link = it.get("link") or ""
            link_part = f" | {link}" if link and not is_google_news(link) else ""
            trimmed.append(f"- {title}{link_part}")

        news_context = "\n".join(trimmed)
        link_instruction = (
            "Include exactly one publisher link on its own line before the hashtags."
            if include_link and any(x.get("link") and not is_google_news(x["link"]) for x in news_items)
            else "Do not include any links."
        )

        persona = (
            "Write like a senior sales leader in tech and fintech who favors partnerships. "
            "Direct, grounded, and human. Confident, not cocky."
        )
        tone_line = (
            "Tone: inspirational, clear call to action."
            if tone == "inspirational"
            else "Tone: thought leadership with one strong point of view and one actionable takeaway."
        )

        prompt = (
            f"{persona} {tone_line} Topic: {topic_key.replace('_', ' ')}.\n\n"
            f"Keep it {words_low} to {words_high} words. Avoid buzzwords. No generic openers like 'As a'. "
            f"No em dashes. No semicolons. Vary sentence length. "
            f"End with two or three relevant hashtags on a separate last line. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )
        return prompt

    # =============== Model calls ===============

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

    def generate_post_with_gemini(self, topic_key, news_items, include_link, tone):
        if not self.gemini_key:
            return None

        attempts = [
            {"api": "v1beta", "model": "gemini-1.5-flash-latest", "keep": 2, "max_out": 520, "words": (120, 170)},
            {"api": "v1beta", "model": "gemini-1.5-flash",        "keep": 2, "max_out": 480, "words": (120, 170)},
            {"api": "v1",     "model": "gemini-2.5-flash",        "keep": 1, "max_out": 600, "words": (120, 170)},
        ]

        headers = {"Content-Type": "application/json"}

        for i, step in enumerate(attempts, start=1):
            base = f"https://generativelanguage.googleapis.com/{step['api']}/models/{step['model']}:generateContent"
            prompt = self._build_prompt(
                topic_key, news_items, include_link, tone,
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
                return self.enforce_style_rules(self.debuzz(text))
            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue
        return None

    def generate_post_with_openai(self, topic_key, news_items, include_link, tone):
        if not self.openai_key:
            return None

        news_context = "\n".join(
            f"- {it['title'][:110]}{'...' if len(it['title'])>110 else ''}"
            f"{' | ' + it['link'] if it.get('link') and not is_google_news(it['link']) else ''}"
            for it in news_items[:2]
        )
        link_instruction = (
            "Include exactly one publisher link on its own line before the hashtags."
            if include_link and any(x.get("link") and not is_google_news(x["link"]) for x in news_items)
            else "Do not include any links."
        )
        tone_line = "inspirational" if tone == "inspirational" else "thought leadership"
        prompt = (
            "Write a LinkedIn post from a senior sales leader in tech and fintech who favors partnerships. "
            f"Tone: {tone_line}. Topic: {topic_key.replace('_', ' ')}.\n\n"
            "Keep it 120 to 170 words. Direct, practical, and human. "
            "Avoid buzzwords and generic openers. No em dashes. No semicolons. "
            "End with two or three relevant hashtags on a separate last line. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            "Return only the post text."
        )

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.65, "max_tokens": 420}
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=45)
            if r.status_code != 200:
                print(f"OpenAI error {r.status_code}: {r.text[:400]}")
                return None
            text = r.json()["choices"][0]["message"]["content"]
            return self.enforce_style_rules(self.debuzz(text))
        except Exception as e:
            print(f"OpenAI call failed: {e}")
            return None

    # Simple polish pass, keeps links and hashtags
    def polish_with_model(self, text: str):
        if not text:
            return text
        prompt = (
            "Edit the LinkedIn post to improve grammar, flow, and clarity. "
            "Keep the same ideas and facts. Do not add or remove URLs. "
            "Keep hashtags at the end on their own line. No em dashes. No semicolons. "
            "Return only the edited text.\n\n"
            f"Post:\n{text}"
        )
        # Prefer Gemini for polish, otherwise OpenAI, else return original
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
                body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 320}}
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"}, json=body, timeout=35)
                if r.status_code == 200:
                    edited = self._extract_text_from_gemini(r.json())
                    if edited:
                        return self.enforce_style_rules(self.debuzz(edited))
            if self.openai_key:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 320}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=35)
                if r.status_code == 200:
                    edited = r.json()["choices"][0]["message"]["content"]
                    if edited:
                        return self.enforce_style_rules(self.debuzz(edited))
        except Exception as e:
            print(f"Polish pass skipped due to error: {e}")
        return self.enforce_style_rules(self.debuzz(text))

    # =============== Local fallback & Quality guard ===============

    def local_compose(self, topic_key: str, include_link: bool, news_items) -> str:
        # Short, useful, senior voice
        hooks = {
            "tech_partnerships": "Partnerships work when both sides bet real resources and chase one measurable outcome.",
            "ai": "AI wins in the enterprise when it reduces time to value, not just cost.",
            "payments": "Payments is a trust business. Speed and acceptance matter, but proof beats promises.",
        }
        hook = hooks.get(topic_key, "Clarity beats noise.")
        bullets = [it["title"] for it in news_items[:2] if it.get("title")]
        insight = (
            f"I’m seeing faster wins when teams anchor on one customer use case, ship a tight integration, "
            f"co sell the first ten deals, and report the revenue impact in the same dashboard."
        )
        close = "Pick one motion, make it repeatable, then scale with the right partners."
        text = f"{hook} {insight} {' '.join(bullets[:1])} {close}"
        text = self.enforce_style_rules(self.debuzz(text))
        # Add link if we can
        if include_link:
            link = self.pick_publisher_link(news_items)
            if link:
                text += f"\n\n{link}"
        return text

    def ensure_hashtags_and_layout(self, text: str, topic_key: str, include_link: bool, news_items) -> str:
        text = self.enforce_style_rules(self.debuzz(text)).strip()

        # Finish sentence if clipped
        if not self.ends_cleanly(text):
            text = text.rstrip(' "\n') + "."

        # Ensure link placement if requested
        if include_link and ("http://" not in text and "https://" not in text):
            good = self.pick_publisher_link(news_items)
            if good:
                # Place link on its own line before hashtags
                text += f"\n\n{good}"

        # Hashtags line
        if not self.has_hashtags(text):
            candidates = self.topic_tags.get(topic_key, ["#partnerships", "#busdev"])
            chosen = []
            for tag in candidates:
                if tag not in chosen:
                    chosen.append(tag)
                if len(chosen) == 3:
                    break
            text += "\n\n" + " ".join(chosen)

        # Final style sweep
        text = self.enforce_style_rules(self.debuzz(text))
        return text

    # =============== LinkedIn ===============

    def post_to_linkedin(self, text: str) -> bool:
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

    # =============== Main run ===============

    def run_weekly_post(self):
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        state = self.load_state()
        if not self.should_post_today(state):
            return

        topic_key, state = self.get_next_topic(state)
        include_link = random.random() < 0.6
        tone = random.choice(["inspirational", "thought_leadership"])

        print(f"--- Topic: {topic_key.replace('_', ' ').title()} | Tone: {tone} | Link: {include_link} ---")
        query = random.choice(self.topics[topic_key])
        print(f"Fetching news for: {query}")
        news_items = self.fetch_news(topic_key, query)
        print(f"Found {len(news_items)} news items")

        # If we wanted a link but didn't get a clean publisher URL, flip to linkless
        if include_link and not any(it.get("link") and not is_google_news(it["link"]) for it in news_items):
            include_link = False
            print("No clean publisher link found, switching to linkless piece.")

        # Generate
        post_text = None
        if self.gemini_key:
            print("Generating post with Gemini...")
            post_text = self.generate_post_with_gemini(topic_key, news_items, include_link, tone)
        if not post_text and self.openai_key:
            print("Gemini failed, using OpenAI...")
            post_text = self.generate_post_with_openai(topic_key, news_items, include_link, tone)
        if not post_text:
            print("All model calls failed, composing locally.")
            post_text = self.local_compose(topic_key, include_link, news_items)

        # Layout and guarantees
        post_text = self.ensure_hashtags_and_layout(post_text, topic_key, include_link, news_items)

        # Polish pass
        post_text = self.polish_with_model(post_text)
        post_text = self.ensure_hashtags_and_layout(post_text, topic_key, include_link, news_items)

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
