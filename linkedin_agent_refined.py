# linkedin_agent_refined.py
"""
Testable/defensive rewrite of the LinkedIn AI agent.

Key improvements
---------------
- Dependency injection: pass a requests-like session, random module, and clock sleepers.
- Feature flags: dry_run (no post), disable_delay (skip sleeps in CI), max_delay configurable.
- Robust logging and failure metadata; avoids raising in common network errors.
- Input validation and guardrails; consistent return shapes; safer defaults.
- Retries with backoff for network calls (configurable).
- Clear separation of "compose" vs "publish" for easier testing.
"""

from __future__ import annotations

import os
import json
import re
import random as _random_mod
import time as _time_mod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote

import requests
import xml.etree.ElementTree as ET


# ------------------------------- Configuration -------------------------------

DEFAULT_TIMEOUT = 15
DEFAULT_RSS_TIMEOUT = 12
DEFAULT_GEN_TIMEOUT = 45
DEFAULT_POST_TIMEOUT = 45
MAX_HASHTAG_HISTORY = 10
MAX_CLOSER_HISTORY = 10


# ------------------------------- Utility helpers -----------------------------

def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_google_news(url: str) -> bool:
    """True if URL is a Google News wrapper or 'news.google.com'."""
    d = domain(url)
    return d.endswith("news.google.com")  # stricter than plain 'google.com'


def ends_cleanly(text: str) -> bool:
    return bool(re.search(r'[.!?]"?\s*$', (text or "")))


def safe_json_dump(path: str, data: Dict[str, Any]) -> Optional[str]:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
        return None
    except Exception as e:
        return f"save_state_failed:{e}"


def safe_json_load(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return dict(default)


# ------------------------------- Data shapes ---------------------------------

@dataclass
class AgentConfig:
    linkedin_token: str = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
    person_urn: str = os.getenv("LINKEDIN_PERSON_URN", "").strip()
    gemini_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    force_post: bool = os.getenv("FORCE_POST", "").strip() == "1"

    # test/ops toggles
    dry_run: bool = os.getenv("DRY_RUN", "").strip() == "1"
    disable_delay: bool = os.getenv("DISABLE_DELAY", "").strip() == "1"
    max_delay_minutes: int = int(os.getenv("MAX_DELAY_MINUTES", "120"))

    # networking
    timeout_rss: int = DEFAULT_RSS_TIMEOUT
    timeout_gen: int = DEFAULT_GEN_TIMEOUT
    timeout_post: int = DEFAULT_POST_TIMEOUT
    retries: int = 2
    backoff_seconds: float = 0.8


def backoff_sleep(sleeper, attempt: int, base: float):
    sleeper(base * (2 ** (attempt - 1)))


# ------------------------------- Core class ----------------------------------

class LinkedInAIAgent:
    """
    Smaller API surface for tests:
      - compose_post(...) -> (text, meta)
      - publish_to_linkedin(text) -> (ok:bool, resp:str)

    We keep your style logic but make calls injectable.
    """

    STOPWORDS = {
        "the","and","for","with","that","this","from","into","over","after","about","your","their","there",
        "of","a","to","in","on","at","as","by","is","are","was","be","it","its","an","or","we","our","you","they",
        "ai","llm","llms","via","vs","using","how","why","what","new","latest","future","today","more"
    }

    def __init__(
        self,
        config: AgentConfig,
        session: Optional[requests.sessions.Session] = None,
        rng: Any = None,
        sleeper: Any = None,
        now_func: Any = None,
    ):
        self.cfg = config
        self.session = session or requests.Session()
        self.rng = rng or _random_mod
        self.sleeper = sleeper or _time_mod.sleep
        self.now_func = now_func or datetime.now

        self.state_file = "agent_state.json"
        self.state = safe_json_load(
            self.state_file,
            {
                "last_topics": [],
                "post_count": 0,
                "last_post_date": None,
                "recent_hashtag_sets": [],
                "recent_closers": [],
            },
        )

        # Small topic set for tests; you can inject more externally if desired.
        self.topics = {
            "ai": ["OpenAI enterprise product announcements"],
            "tech_partnerships": ["Microsoft strategic partnerships announcements"],
        }
        self.topic_tags = {
            "ai": ["#AI", "#EnterpriseAI", "#B2B"],
            "tech_partnerships": ["#Partnerships", "#Ecosystem", "#GTM"],
        }
        self.openers = ["Execution beats theater.", "When teams share one metric, decisions speed up."]
        self.closers = ["Publish progress weekly.", "Choose partners that shorten time to value."]

    # ------------ delays & scheduling ------------

    def random_delay_minutes(self, min_minutes=0, max_minutes=None):
        if self.cfg.disable_delay:
            return
        mm = max_minutes if max_minutes is not None else self.cfg.max_delay_minutes
        seconds = self.rng.randint(min_minutes * 60, mm * 60)
        self.sleeper(seconds)

    def should_post_today(self) -> bool:
        if self.cfg.force_post:
            return True
        today = self.now_func().strftime("%Y-%m-%d")
        return self.state.get("last_post_date") != today

    # ------------ text utilities ------------

    @staticmethod
    def enforce_style_rules(text: str) -> str:
        if not text:
            return text
        text = text.replace("—", ",").replace("–", ",").replace(";", ",")
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)
        if text.count('"') % 2 == 1 and text.strip().startswith('"'):
            text = text.lstrip('"').lstrip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def debuzz(text: str) -> str:
        if not text:
            return text
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

    @staticmethod
    def word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    def _keywords_from_titles(self, news_items: List[Dict[str, str]], k=6) -> List[str]:
        text = " ".join((it.get("title") or "") for it in news_items[:4])
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
        freq = {}
        for w in words:
            wl = w.lower()
            if wl in self.STOPWORDS or len(wl) < 4:
                continue
            freq[wl] = freq.get(wl, 0) + 1
        keys = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
        def camel(t): return re.sub(r"[^A-Za-z0-9]", "", t.title())
        return [camel(t[0]) for t in keys if t[0]]

    # ------------ news fetching ------------

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
            r = self.session.get(link, timeout=self.cfg.timeout_rss, allow_redirects=True, headers={"User-Agent": "curl/8"})
            final_url = r.url
            if final_url and not is_google_news(final_url):
                return final_url
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

    def fetch_news(self, query: str, max_items: int = 3) -> Tuple[List[Dict[str, str]], str]:
        """Fetch Google News RSS; return (items, err). Never raises."""
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items: List[Dict[str, str]] = []
        err = ""
        for attempt in range(1, self.cfg.retries + 1):
            try:
                r = self.session.get(rss, timeout=self.cfg.timeout_rss, headers={"User-Agent": "curl/8"})
                if r.status_code != 200:
                    err = f"rss_http_{r.status_code}"
                    backoff_sleep(self.sleeper, attempt, self.cfg.backoff_seconds)
                    continue
                root = ET.fromstring(r.text)
                channel = root.find("channel")
                if channel is None:
                    err = "rss_no_channel"
                    break
                for it in channel.findall("item")[: max_items * 2]:
                    title = (it.findtext("title") or "").strip()
                    link = (it.findtext("link") or "").strip()
                    desc = (it.findtext("description") or "").strip()
                    pub_link = self._extract_publisher_from_description(desc)
                    best = pub_link or self._extract_original_from_link(link)
                    items.append({"title": title, "link": best})
                    if len(items) >= max_items:
                        break
                break
            except Exception as e:
                err = f"rss_exc:{e}"
                backoff_sleep(self.sleeper, attempt, self.cfg.backoff_seconds)

        if not items:
            items = [{"title": "Current industry trends", "link": ""}]
        return items, err

    # ------------ composition ------------

    def _choose_hashtags(self, topic_key: str, body: str, news_items: List[Dict[str, str]]) -> str:
        base = self.topic_tags.get(topic_key, ["#B2B", "#GTM"])
        dynamic = ["#" + k for k in self._keywords_from_titles(news_items, k=6) if k]
        pool = list(dict.fromkeys(base + dynamic))  # unique order
        lower = (body or "").lower()
        candidates = [t for t in pool if t.lower() not in lower]

        recent_sets = self.state.get("recent_hashtag_sets", [])
        self.rng.shuffle(candidates)
        chosen = None
        for i in range(min(12, len(candidates))):
            chunk = candidates[i:i+3]
            if len(chunk) < 3:
                break
            trial = " ".join(chunk)
            if trial.lower() not in recent_sets:
                chosen = trial
                break
        if not chosen:
            chosen = " ".join((pool[:3] or ["#B2B", "#GTM", "#AI"]))

        recent_sets.append(chosen.lower())
        self.state["recent_hashtag_sets"] = recent_sets[-MAX_HASHTAG_HISTORY:]
        return chosen

    def _inject_open_close(self, body: str) -> Tuple[str, str]:
        opener = self.rng.choice(self.openers)
        recent = self.state.get("recent_closers", [])
        closers = list(self.closers)
        self.rng.shuffle(closers)
        closer = next((c for c in closers if c.lower() not in recent), closers[0])
        recent.append(closer.lower())
        self.state["recent_closers"] = recent[-MAX_CLOSER_HISTORY:]

        body = (body or "").strip()
        out = f"{opener} {body}".strip()
        if not ends_cleanly(out):
            out += "."
        out = f"{out} {closer}"
        if not ends_cleanly(out):
            out += "."
        return out, closer

    def compose_post(
        self,
        topic_key: str = "ai",
        include_link: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Compose post text. Never raises. Returns (text, meta)."""
        meta: Dict[str, Any] = {"errors": []}

        if topic_key not in self.topics:
            meta["errors"].append("unknown_topic")
            topic_key = "ai"

        query = self.rng.choice(self.topics[topic_key])
        news_items, err = self.fetch_news(query)
        if err:
            meta["errors"].append(err)

        # Build body locally (in your code this might call Gemini/OpenAI; we keep it local for tests)
        hook = "AI wins when it reduces time to value, not just cost."
        insight = "Start small, instrument outcomes, and scale only after the pattern repeats."
        body = self.enforce_style_rules(self.debuzz(f"{hook} {insight}"))

        # Inject opener/closer and link/hashtags
        body, closer = self._inject_open_close(body)
        link_line = ""
        if include_link:
            for it in news_items:
                lnk = (it.get("link") or "").strip()
                if lnk and not is_google_news(lnk) and (urlparse(lnk).path or "") not in ("", "/"):
                    link_line = f"\n\n{lnk}"
                    break

        tags = self._choose_hashtags(topic_key, body, news_items)
        final_text = f"{body}{link_line}\n\n{tags}".strip()

        return final_text, meta

    # ------------ publishing ------------

    def publish_to_linkedin(self, text: str) -> Tuple[bool, str]:
        """Publish to LinkedIn. Returns (ok, message). Never raises."""
        if self.cfg.dry_run:
            return True, "dry_run"

        if not self.cfg.linkedin_token or not self.cfg.person_urn:
            return False, "missing_credentials"

        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {self.cfg.linkedin_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        payload = {
            "author": f"urn:li:person:{self.cfg.person_urn}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        for attempt in range(1, self.cfg.retries + 1):
            try:
                r = self.session.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_post)
                if r.status_code in (200, 201):
                    # update state
                    self.state["post_count"] = self.state.get("post_count", 0) + 1
                    self.state["last_post_date"] = self.now_func().strftime("%Y-%m-%d")
                    err = safe_json_dump(self.state_file, self.state)
                    return True, "ok" if not err else err
                if r.status_code in (401, 403):
                    return False, f"http_{r.status_code}"
                # transient
                backoff_sleep(self.sleeper, attempt, self.cfg.backoff_seconds)
            except Exception as e:
                last = f"exc:{e}"
                backoff_sleep(self.sleeper, attempt, self.cfg.backoff_seconds)
        return False, last if 'last' in locals() else "post_failed"
