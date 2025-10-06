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
        # Remove generic openers
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)
        # Trim unmatched opening quote
        if text.count('"') % 2 == 1 and text.strip().startswith('"'):
            text = text.lstrip('"').lstrip()
        return text.strip()

    def has_hashtags(self, text):
        return len(re.findall(r"(?:^|\s)#\w+", text)) >= 2

    def ends_cleanly(self, text):
        return bool(re.search(r'[.!?]"?\s*$', text))

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
        try:
            if not link:
                return ""
            parsed = urlparse(link)
            if "news.google.com" not in parsed.netloc:
                return link
            q = parse_qs(parsed.query)
            if "url" in q and q["url"]:
                return unquote(q["url"][0])
            try:
                r = requests.get(link, timeout=10, allow_redirects=True, headers={"User-Agent": "curl/8"})
                final_url = r.url
                if final_url and "news.google.com" not in urlparse(final_url).netloc:
                    return final_url
            except Exception:
                pass
            return link
        except Exception:
            return link

    def _extract_publisher_from_description(self, description):
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

    # -----------------------------
    # Prompt helpers
    # -----------------------------
    def _build_prompt(self, topic_key, news_items, include_link, keep=2, words_low=110, words_high=150):
        trimmed = []
        for it in news_items[:keep]:
            title = it["title"][:110] + ("..." if len(it["title"]) > 110 else "")
            link = it.get("link") or ""
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
            f"Avoid generic set ups like 'As a'. No em dashes. No semicolons. "
            f"End with two or three relevant hashtags. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )
        return prompt

    # -----------------------------
    # Gemini generation
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
            {"api": "v1beta", "model": "gemini-1.5-flash-latest", "keep": 2, "max_out": 520, "words": (110, 150)},
            {"api": "v1beta", "model": "gemini-1.5-flash",        "keep": 2, "max_out": 480, "words": (110, 150)},
            {"api": "v1",     "model": "gemini-2.5-flash",        "keep": 1, "max_out": 600, "words": (110, 150)},
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

                return self.enforce_style_rules(text)

            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue

        return None

    # -----------------------------
    # OpenAI fallback
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
            f"Keep it 110 to 150 words. Be direct, practical, and human. "
            f"Avoid generic set ups like 'As a'. No em dashes. No semicolons. "
            f"End with two or three relevant hashtags. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 420}
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=45)
            if r.status_code != 200:
                print(f"OpenAI error {r.status_code}: {r.text[:400]}")
                return None
            text = r.json()["choices"][0]["message"]["content"]
            return self.enforce_style_rules(text)
        except Exception as e:
            print(f"OpenAI call failed: {e}")
            return None

    # -----------------------------
    # Local finisher and quality guard
    # -----------------------------
    def pick_publisher_link(self, news_items):
        for it in news_items:
            link = it.get("link") or ""
            if link and "news.google.com" not in link:
                return link
        return ""

    def local_finish(self, partial, include_link, news_items):
        # Finish cleanly with a short closer and hashtags
        closer = "The teams that win pick one motion, ship fast, measure impact, then scale."
        if not self.ends_cleanly(partial):
            partial = partial.rstrip(' "\n') + "."
        text = partial + " " + closer
        if not self.has_hashtags(text):
            text += "\n\n#partnerships #busdev #fintech"
        if include_link and ("http://" not in text and "https://" not in text):
            good = self.pick_publisher_link(news_items)
            if good:
                text += f"\n\n{good}"
        return self.enforce_style_rules(text)

    def post_quality_guard(self, text, include_link, news_items):
        if not text:
            return None

        text = self.enforce_style_rules(text)

        # Too short or ends mid thought
        if len(text) < 120 or not self.ends_cleanly(text):
            return self.local_finish(text, include_link, news_items)

        # Missing hashtags
        if not self.has_hashtags(text):
            text += "\n\n#partnerships #busdev #fintech"

        # Link handling
        if include_link and ("http://" not in text and "https://" not in text):
            good = self.pick_publisher_link(news_items)
            if good:
