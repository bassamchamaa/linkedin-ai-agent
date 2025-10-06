import os
import json
import re
import random
from datetime import datetime
import requests


class LinkedInAIAgent:
    def __init__(self):
        # Secrets
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        # Topic rotation (tech partnerships, AI, payments)
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
        if not self.gemini_key:
            print("Gemini key missing. Set GEMINI_API_KEY.")

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

    def enforce_style_rules(self, text):
        # No em or en dashes
        text = text.replace("—", ",").replace("–", ",")
        return text.strip()

    # -----------------------------
    # News fetch via Google News RSS
    # -----------------------------
    def fetch_news(self, topic_key, query, max_items=6):
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items = []
        try:
            r = requests.get(rss, timeout=12)
            if r.status_code != 200:
                print(f"News fetch error {r.status_code} for {topic_key}: {r.text[:200]}")
                return items

            content = r.text
            titles_raw = re.findall(r"<title>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</title>", content, flags=re.I | re.S)
            titles = [(a or b) for a, b in titles_raw]
            links = re.findall(r"<link>(.*?)</link>", content, flags=re.I | re.S)

            # Skip the feed title
            paired = list(zip(titles[1:], links[1:]))[:max_items]
            for title, link in paired:
                # Unwrap Google redirect
                if "news.google.com" in link and "url=" in link:
                    m = re.search(r"[?&]url=([^&]+)", link)
                    if m:
                        link = requests.utils.unquote(m.group(1))
                items.append({"title": title.strip(), "link": link.strip()})
        except Exception as e:
            print(f"Error fetching news for {topic_key}: {e}")

        if not items:
            print(f"⚠ No news found for {topic_key}, using fallback.")
            items = [{"title": f"Current industry trends in {topic_key.replace('_', ' ')}", "link": ""}]
        return items

    # -----------------------------
    # Gemini helpers
    # -----------------------------
    def _extract_text_from_gemini(self, payload):
        """
        Robustly extract text across response shapes. Returns None if empty.
        """
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

    def _models_available(self, api_version):
        try:
            url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={self.gemini_key}"
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                return {}
            models = r.json().get("models", [])
            return {m.get("name", "").split("/")[-1]: m for m in models}
        except Exception:
            return {}

    def _build_prompt(self, topic_key, news_items, include_link, keep=2, words_low=120, words_high=160):
        # Lean prompt to avoid MAX_TOKENS
        trimmed = []
        for it in news_items[:keep]:
            title = it["title"]
            if len(title) > 110:
                title = title[:107] + "..."
            link_part = f" | {it['link']}" if it.get("link") else ""
            trimmed.append(f"- {title}{link_part}")

        news_context = "\n".join(trimmed)
        link_instruction = (
            "Include exactly one news link in the body."
            if include_link and any(x.get("link") for x in news_items)
            else "Do not include any links."
        )

        prompt = (
            f"Write a LinkedIn post from a senior tech and fintech partnerships leader about "
            f"{topic_key.replace('_', ' ')}.\n\n"
            f"Keep it {words_low} to {words_high} words. Make it concrete, strategic, and useful for BD and GTM leaders. "
            f"Do not use em dashes. Add two or three relevant hashtags at the end. "
            f"{link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            f"Return only the post text."
        )
        return prompt

    def generate_post_with_gemini(self, topic_key, news_items, include_link):
        """
        Retry strategy on v1beta only with tight prompts.
        Removes unsupported fields like responseMimeType, systemInstruction, thinking.
        """
        attempts = [
            {"api": "v1beta", "model": "gemini-1.5-flash-latest", "keep": 2, "max_out": 520, "words": (120, 160)},
            {"api": "v1beta", "model": "gemini-1.5-flash", "keep": 2, "max_out": 480, "words": (110, 150)},
            {"api": "v1beta", "model": "gemini-1.5-pro-latest", "keep": 1, "max_out": 480, "words": (110, 150)},
            {"api": "v1beta", "model": "gemini-1.5-pro", "keep": 1, "max_out": 440, "words": (100, 140)},
        ]

        # Filter to models your key actually has
        cache = {}
        filtered = []
        for a in attempts:
            if a["api"] not in cache:
                cache[a["api"]] = self._models_available(a["api"])
            if a["model"] in cache[a["api"]]:
                filtered.append(a)
        if not filtered:
            print("No suitable Gemini models available from ListModels.")
            return None

        headers = {"Content-Type": "application/json"}

        for i, step in enumerate(filtered, start=1):
            base = (
                f"https://generativelanguage.googleapis.com/"
                f"{step['api']}/models/{step['model']}:generateContent"
            )
            prompt = self._build_prompt(
                topic_key,
                news_items,
                include_link,
                keep=step["keep"],
                words_low=step["words"][0],
                words_high=step["words"][1],
            )
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": step["max_out"],
                },
            }

            print(f"Attempt {i}: {step['model']} on {step['api']} with maxOutputTokens={step['max_out']}")
            try:
                resp = requests.post(f"{base}?key={self.gemini_key}", headers=headers, json=body, timeout=45)
                if resp.status_code != 200:
                    print(f"Gemini error {resp.status_code} on attempt {i}: {resp.text[:600]}")
                    continue

                data = resp.json()
                text = self._extract_text_from_gemini(data)
                if not text:
                    # If finishReason was MAX_TOKENS and no text was surfaced, try next attempt
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

                if include_link:
                    has_link = "http://" in text or "https://" in text
                    if not has_link:
                        for it in news_items:
                            if it.get("link"):
                                text += f"\n\n{it['link']}"
                                break
                        text = self.enforce_style_rules(text)

                return text

            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue

        return None

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
        post_type = "with reference link" if include_link else "thought leadership piece"

        print(f"--- Topic: {topic_key.replace('_', ' ').title()} ({post_type}) ---")
        query = random.choice(self.topics[topic_key])
        print(f"Fetching news for: {query}")
        news_items = self.fetch_news(topic_key, query)
        print(f"Found {len(news_items)} news items")

        if not self.gemini_key:
            print("No Gemini key set, cannot generate content.")
            return

        print("Generating post with Gemini...")
        post_text = self.generate_post_with_gemini(topic_key, news_items, include_link)
        if not post_text:
            print("⚠ Post generation returned None after retries.")
            return

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
