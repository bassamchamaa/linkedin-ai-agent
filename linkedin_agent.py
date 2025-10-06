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
                "ecosystem partnerships SaaS"
            ],
            "ai": [
                "artificial intelligence enterprise",
                "AI business applications",
                "machine learning partnerships",
                "AI go to market"
            ],
            "payments": [
                "fintech payments partnerships",
                "payment technology trends",
                "digital payments innovation",
                "real time payments B2B"
            ],
            # NEW TOPICS
            "agentic_commerce": [
                "agentic commerce",
                "AI shopping agents ecommerce",
                "autonomous agents retail",
                "agentic AI marketplaces"
            ],
            "generative_ai": [
                "generative AI enterprise",
                "gen AI business applications",
                "foundation models partnerships",
                "LLM adoption B2B"
            ],
            "ai_automations": [
                "AI automation enterprise",
                "workflow automation AI",
                "RPA and AI partnerships",
                "AI agents for operations"
            ],
        }

        # Hashtag pools by topic
        self.topic_tags = {
            "tech_partnerships": ["#partnerships", "#busdev", "#GTM", "#SaaS"],
            "ai": ["#AI", "#EnterpriseAI", "#B2B", "#GTM"],
            "payments": ["#fintech", "#payments", "#partnerships", "#B2B"],
            "agentic_commerce": ["#AgenticAI", "#ecommerce", "#retailtech"],
            "generative_ai": ["#GenAI", "#EnterpriseAI", "#LLM"],
            "ai_automations": ["#automation", "#AIagents", "#operations"],
        }

        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("LinkedIn secrets missing. Check LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key and not self.openai_key:
            print("No model key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")

    # =============== Style & Quality ===============

    def enforce_style_rules(self, text: str) -> str:
        # No em or en dashes, no semicolons
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
           
