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

        # Gemini endpoint cache (set by resolve_gemini_endpoint)
        self.gemini_api_version = None   # "v1" or "v1beta"
        self.gemini_model = None         # e.g. "gemini-1.5-flash-latest"

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

        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("LinkedIn secrets missing. Check LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key:
            print("Gemini key missing. Set GEMINI_API_KEY.")

    # -----------------------------
    # Utilities
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
        text = text.replace("—", ",").replace("–", ",")
        return text.strip()

    # -----------------------------
    # News fetching (Google News RSS)
    # -----------------------------
    def fetch_news(self, topic_key, query):
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items = []
        try:
            r = requests.get(rss, timeout=12)
            if r.status_code != 200:
                print(f"News fetch error {r.status_code} for {topic_key}: {r.text[:200]}")
                return items

            content = r.text
            titles_raw = re.findall(
                r"<title>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</title>", content, flags=re.I | re.S
            )
            titles = [(a or b) for a, b in titles_raw]
            links = re.findall(r"<link>(.*?)</link>", content, flags=re.I | re.S)
            paired = list(zip(titles[1:], links[1:]))

            for title, link in paired[:6]:
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
    # Gemini endpoint resolution
    # -----------------------------
    def resolve_gemini_endpoint(self):
        """
        Find a working model + API version for this key by listing models.
        Caches the result on the instance.
        """
        if self.gemini_api_version and self.gemini_model:
            return

        prefs = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest",
        ]

        for api_version in ["v1", "v1beta"]:
            try:
                url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={self.gemini_key}"
                r = requests.get(url, timeout=12)
                if r.status_code != 200:
                    # Try next version
                    continue
                models = r.json().get("models", [])
                available = {m.get("name", "").split("/")[-1]: m for m in models}

                # Choose preferred available model that supports generateContent if present
                chosen = None
                for name in prefs:
                    m = available.get(name)
                    if not m:
                        continue
                    methods = m.get("supportedGenerationMethods", [])
                    if not methods or "generateContent" in methods:
                        chosen = name
                        break

                # Fallback to any model that includes gemini and supports generateContent
                if not chosen:
                    for name, m in available.items():
                        methods = m.get("supportedGenerationMethods", [])
                        if "gemini" in name and ("generateContent" in methods or not methods):
                            chosen = name
                            break

                if chosen:
                    self.gemini_api_version = api_version
                    self.gemini_model = chosen
                    print(f"Using Gemini model: {chosen} on {api_version}")
                    return
            except Exception as e:
                # Move on to the next version
                print(f"Model list error on {api_version}: {e}")
                continue

        # If everything fails, default to v1 + gemini-pro and let the API error guide us
        self.gemini_api_version = "v1"
        self.gemini_model = "gemini-pro"
        print("Falling back to Gemini model: gemini-pro on v1")

    # -----------------------------
    # Gemini generation
    # -----------------------------
    def generate_post_with_gemini(self, topic_key, news_items, include_link):
        self.resolve_gemini_endpoint()

        news_context = "\n".join([f"- {i['title']}: {i['link']}" for i in news_items[:3]])
        link_instruction = (
            "You MUST include exactly one link to one of the news articles in the body of the post."
            if include_link and any(x.get("link") for x in news_items)
            else "Do NOT include any links. This should be a thought leadership piece based on current trends."
        )

        prompt = f"""You are a senior sales leader in tech and fintech with deep expertise in strategic partnerships. Create a LinkedIn post about {topic_key.replace('_', ' ')}.

Recent news and trends:
{news_context}

Requirements:
- 150 to 220 words
- Write in the voice of a senior partnerships and revenue leader
- Focus on partnerships, deal-making, or GTM strategy for tech and fintech
- {link_instruction}
- Add 2 or 3 relevant hashtags at the end
- Make it actionable, with a clear point of view
- Do not use em dashes. Use commas, periods, or colons instead.
- Return only the post text, ready to publish.
"""

        base = f"https://generativelanguage.googleapis.com/{self.gemini_api_version}/models/{self.gemini_model}:generateContent"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 600},
        }

        try:
            resp = requests.post(f"{base}?key={self.gemini_key}", headers=headers, json=body, timeout=45)
            if resp.status_code != 200:
                print(f"Gemini error {resp.status_code}: {resp.text[:400]}")
                return None
            data = resp.json()
            post = data["candidates"][0]["content"]["parts"][0]["text"]
            post = self.enforce_style_rules(post)

            if include_link:
                has_link = "http://" in post or "https://" in post
                if not has_link:
                    for it in news_items:
                        if it.get("link"):
                            post += f"\n\n{it['link']}"
                            break
                    post = self.enforce_style_rules(post)
            return post
        except Exception as e:
            print(f"Error generating post: {e}")
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
                print("Hint: 401 usually means the access token is expired or missing w_member_social scope.")
            if r.status_code == 403:
                print("Hint: 403 can mean the app is not approved for member posts or the token lacks permissions.")
            return False
        except Exception as e:
            print(f"✗ Error posting to LinkedIn: {e}")
            return False

    # -----------------------------
    # Main
    # -----------------------------
    def run_weekly_post(self):
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        state = self.load_state()
        if not self.should_post_today(state):
            return

        topic_key, state = self.get_next_topic(state)
        include_link = random.random() < 0.60
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
            print("⚠ Post generation returned None. See any Gemini error above.")
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
            print(f"Next topics in rotation window: {[t for t in self.topics if t not in state['last_topics']]}")
            print(f"Posts remaining in cycle window: {remaining}")
        else:
            print("\n❌ Failed to publish post")


if __name__ == "__main__":
    agent = LinkedInAIAgent()
    agent.run_weekly_post()
