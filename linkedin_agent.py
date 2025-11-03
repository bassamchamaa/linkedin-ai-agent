#!/usr/bin/env python3
"""
LinkedIn AI Agent
=================

Generates and posts LinkedIn updates on a defined cadence for a
"senior tech/fintech sales leader" persona. Posts must:

* Land between 100 and 150 words.
* Rotate across predefined topic buckets and prompt structures.
* Prefer PYMNTS.com coverage for payments topics, otherwise fall back to
  Google News RSS with publisher link extraction.
* End with a clean publisher link (when available) and exactly three
  curated hashtags on the final line.

The agent supports Gemini 2.5 Flash and GPT-5 Nano models, enforces
style constraints, and stores lightweight state in ``agent_state.json``.
"""

import os
import json
import re
import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
import xml.etree.ElementTree as ET
import requests
import time
from typing import Tuple, List, Dict
from html import unescape  # NEW

# -------------------------------
# Configuration
# -------------------------------
WORD_MIN = 100
WORD_MAX = 150

# Robust final-line hashtag matcher (exactly one line, >=3 tags)
HASHTAG_BLOCK_RE = re.compile(r"^(?:#\w+(?:\s+#\w+){2,})$", re.IGNORECASE)


# -------------------------------
# Utility: randomized post delay
# -------------------------------
def random_delay_minutes(min_minutes: int = 0, max_minutes: int = 120) -> None:
    """Sleep for a random amount of time within the specified range unless SKIP_DELAY=1 or DISABLE_DELAY=1."""
    if os.getenv("SKIP_DELAY") == "1" or os.getenv("DISABLE_DELAY") == "1":
        print("Delay disabled (SKIP_DELAY/DISABLE_DELAY set) — skipping randomized wait.")
        return
    delay_seconds = random.randint(min_minutes * 60, max_minutes * 60)
    delay_minutes = delay_seconds / 60
    print(f"Random delay: waiting {delay_minutes:.1f} minutes before posting...")
    time.sleep(delay_seconds)


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_google_news(url: str) -> bool:
    d = domain(url)
    return d.endswith("news.google.com") or d.endswith("google.com")


# --------------------------------
# Constants for style/variation
# --------------------------------
INSIGHT_VARIATIONS = [
    "Make the metric visible to both teams and review it weekly. Public scoreboards change behavior.",
    "Reduce the first value moment. Shorter time-to-value beats feature depth when budgets are tight.",
    "Instrument the handoffs. Most friction hides between systems, not within them.",
    "Prove the pattern in one segment, then clone it with the same playbook and safeguards.",
]


# --------------------------------
# HTML meta extractor (for PYMNTS)
# --------------------------------
def _extract_meta(html: str, name: str) -> str:
    """Pulls <meta property/name=... content=...> or falls back to <title> for og/twitter title."""
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            return unescape(m.group(1)).strip()
    if name in ("og:title", "twitter:title"):
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    return ""


# -------------------------------
# Main Agent
# -------------------------------
class LinkedInAIAgent:
    # Hashtag stopwords + brand bans
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "into", "over", "after",
        "about", "your", "their", "there", "of", "a", "to", "in", "on", "at", "as", "by",
        "is", "are", "was", "be", "it", "its", "an", "or", "we", "our", "you", "they",
        "ai", "llm", "llms", "via", "vs", "using", "how", "why", "what", "new", "latest",
        "future", "today", "more"
    }
    BANNED_BRAND_TAGS = {"#paypal", "#visa", "#mastercard", "#stripe", "#adyen"}

    def __init__(self) -> None:
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.force_post = os.getenv("FORCE_POST", "").strip() == "1"
        self.news_source = os.getenv("NEWS_SOURCE", "").strip().lower()  # NEW

        # ---------- Topics & queries ----------
        self.topics: Dict[str, List[str]] = {
            "tech_partnerships": [
                "Microsoft strategic partnerships announcements",
                "Oracle enterprise alliance deals",
                "SAP ecosystem partner programs",
                "IBM business partnership strategy",
                "Cisco partner ecosystem expansion",
                "Salesforce partnership announcements",
                "Adobe partner integrations enterprise",
                "Amazon AWS partner network",
                "Nvidia enterprise AI product launch",
                "Microsoft Azure enterprise strategy",
                "Google Cloud enterprise adoption",
                "Oracle cloud enterprise transformation",
                "SAP enterprise software innovation",
                "Broadcom enterprise networking strategy",
                "Microsoft enterprise collaboration future",
                "Meta enterprise productivity tools",
                "Tencent enterprise digital workplace",
                "Samsung enterprise mobility strategy",
            ],
            "ai": [
                "OpenAI enterprise product announcements",
                "Anthropic Claude enterprise adoption",
                "Microsoft Copilot enterprise rollout",
                "Google Gemini business strategy",
                "Nvidia AI enterprise platform",
                "Databricks AI enterprise launch",
                "Hugging Face enterprise AI models",
                "Cohere enterprise AI deployment",
                "Mistral AI enterprise offerings",
                "Scale AI enterprise platform",
                "OpenAI enterprise partnerships",
                "Anthropic strategic alliances",
                "Microsoft AI partnerships",
                "Amazon AI partnerships deals",
                "Meta AI business partnerships",
                "IBM Watson AI partnerships",
                "Oracle AI integration partnerships",
                "OpenAI enterprise market impact",
                "Anthropic enterprise AI strategy",
                "Replit enterprise developer tools",
                "Cursor AI developer productivity",
                "Databricks AI enterprise transformation",
                "Cerebras AI infrastructure enterprise",
                "Runway AI enterprise creative tools",
                "AI agents enterprise productivity",
                "OpenAI workplace transformation",
                "Microsoft AI future of work",
                "Google AI enterprise automation",
                "Anthropic AI business operations",
                "Adept AI enterprise workflow",
                "Harvey AI enterprise legal work",
                "Glean AI enterprise search",
                "Character AI enterprise applications",
            ],
            "payments": [
                "Stripe strategic partnerships fintech",
                "Visa payment ecosystem partnerships",
                "Mastercard fintech alliance deals",
                "Adyen merchant partnerships",
                "PayPal business partnership strategy",
                "Block Square merchant partnerships",
                "Fiserv payment partnerships",
                "Stripe product launch enterprise",
                "Visa payment innovation enterprise",
                "Mastercard digital payment strategy",
                "Adyen unified commerce platform",
                "PayPal enterprise payment solutions",
                "Checkout.com enterprise payments",
                "Worldpay enterprise merchant services",
                "FIS payment technology enterprise",
                "Stripe embedded finance enterprise",
                "Plaid financial data connectivity",
                "Brex corporate spend management",
                "Revolut business banking innovation",
                "Nubank digital banking expansion",
                "Wise cross-border payments enterprise",
                "Coinbase enterprise crypto adoption",
                "embedded payments enterprise strategy",
                "payment orchestration enterprise future",
                "digital wallet enterprise adoption",
                "real-time payments business impact",
                "Stripe payment infrastructure future",
                "Visa payment innovation future",
            ],
            "agentic_commerce": [
                "AI shopping agents ecommerce launch",
                "autonomous commerce platform enterprise",
                "AI checkout optimization enterprise",
                "conversational commerce AI enterprise",
                "Amazon AI commerce partnerships",
                "Shopify AI merchant partnerships",
                "Adobe commerce AI integration",
                "Salesforce commerce AI partnerships",
                "AI agents ecommerce transformation",
                "autonomous shopping enterprise impact",
                "AI commerce personalization enterprise",
                "conversational AI retail strategy",
                "agentic commerce future retail",
                "AI shopping assistants enterprise",
                "autonomous checkout future",
                "AI commerce orchestration future",
            ],
            "generative_ai": [
                "OpenAI GPT enterprise launch",
                "Anthropic Claude business release",
                "Microsoft Copilot enterprise update",
                "Google Gemini enterprise features",
                "Amazon Bedrock enterprise AI",
                "Meta Llama enterprise deployment",
                "Mistral AI enterprise models",
                "Cohere enterprise AI platform",
                "Stability AI enterprise solutions",
                "ElevenLabs enterprise voice AI",
                "generative AI enterprise governance",
                "AI safety enterprise deployment",
                "LLM enterprise risk management",
                "AI model governance enterprise",
                "responsible AI enterprise strategy",
                "generative AI enterprise adoption trends",
                "LLM business transformation impact",
                "AI content generation enterprise",
                "generative AI developer productivity",
                "AI coding assistants enterprise impact",
                "generative AI future of work",
                "LLM enterprise automation future",
                "AI copilots workplace transformation",
                "generative AI business model innovation",
            ],
            "ai_automations": [
                "AI workflow automation enterprise launch",
                "intelligent process automation enterprise",
                "AI agent orchestration platform",
                "business process AI enterprise",
                "Databricks workflow automation enterprise",
                "IBM automation AI platform",
                "AI automation strategic partnerships",
                "RPA AI integration partnerships",
                "workflow automation vendor alliances",
                "Microsoft automation partnerships",
                "SAP intelligent automation partnerships",
                "AI agents business process transformation",
                "intelligent automation ROI enterprise",
                "AI workforce augmentation impact",
                "AI automation future of work",
                "autonomous business operations future",
                "AI workforce collaboration future",
                "intelligent automation enterprise trends",
                "AI process optimization future",
            ],
        }

        # Curated hashtag pools per topic
        self.topic_tags: Dict[str, List[str]] = {
            "tech_partnerships": ["#Partnerships", "#Ecosystem", "#GTM", "#B2B", "#SaaS"],
            "ai": ["#AI", "#EnterpriseAI", "#B2B", "#GTM", "#ML"],
            "payments": ["#Fintech", "#Payments", "#EmbeddedFinance", "#Risk", "#B2B"],
            "agentic_commerce": ["#AgenticAI", "#Ecommerce", "#CX", "#Conversion"],
            "generative_ai": ["#GenerativeAI", "#LLMs", "#EnterpriseAI", "#Product"],
            "ai_automations": ["#Automation", "#AIAgents", "#Ops", "#Productivity"],
        }

        # Style modes + openers/closers banks to rotate
        self.style_modes = ["story", "tactical", "contrarian", "data_point", "playbook"]
        self.openers = [
            "The fastest wins in enterprise come from reducing time to value.",
            "Great partnerships feel simple to customers because the hard work is invisible.",
            "When teams share one metric, decisions speed up.",
            "Execution beats theater. Especially in B2B.",
            "The best AI outcomes start with clear owners and clean data.",
            "Partnerships work when the field asks for them, not when slides announce them.",
        ]
        self.closers = [
            "Pick a single metric, align the incentives, and publish progress weekly.",
            "Start small, prove value, then scale the playbook with the right partners.",
            "Make it measurable, repeatable, and boring—in the best way.",
            "One win at a time, instrumented end to end.",
            "Shift meetings into dashboards and let the results speak.",
            "Choose partners that shorten the path to customer value.",
        ]

        # State & memory
        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("Missing LinkedIn secrets. Set LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key and not self.openai_key:
            print("No model key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")

    # -------------------------------
    # Style & sanitation helpers
    # -------------------------------
    def enforce_style_rules(self, text: str) -> str:
        if not text:
            return text

        # normalize punctuation
        text = text.replace("—", ",").replace("–", ",").replace(";", ",")

        # remove "As a ..." opener
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)

        # ensure single space after punctuation, and tidy spaces
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\s*!\s*", "! ", text)
        text = re.sub(r"\s*\?\s*", "? ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([,!.?])", r"\1", text)

        # strip stray opening quote if unmatched
        if text.count('"') % 2 == 1 and text.strip().startswith('"'):
            text = text.lstrip('"').lstrip()

        return text.strip()

    def strip_model_wrappers(self, text: str) -> str:
        """Remove meta prefaces like 'Here’s the refined post', code fences, 'Draft:', etc."""
        if not text:
            return text

        # code fences
        text = re.sub(r"^```(\w+)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

        # common prefaces
        preface_patterns = [
            r"^\s*here('?|)s (the )?(refined|final|edited|updated) (linkedin )?post[:\-–]\s*",
            r"^\s*(below is|here is) (the )?(final|refined|edited|updated) (linkedin )?post[:\-–]\s*",
            r"^\s*(draft|final|output|rewrite|re[- ]written|polished)[:\-–]\s*",
            r"^\s*(linked ?in|linkedin) (post|draft)[:\-–]\s*",
        ]
        for pat in preface_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

        # drop first paragraph if clearly meta
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if paras and re.search(r"\b(refined|expanded|polished|rewrite|edited)\b", paras[0], flags=re.IGNORECASE):
            paras = paras[1:] or paras
        text = "\n\n".join(paras).strip()
        return text

    def debuzz(self, text: str) -> str:
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

    def ends_cleanly(self, text: str) -> bool:
        return bool(re.search(r'[.!?]"?\s*$', text))

    def word_count(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    # -------------------------------
    # Persistence
    # -------------------------------
    def load_state(self) -> dict:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "last_topics": [],
            "post_count": 0,
            "last_post_date": None,
            "recent_hashtag_sets": [],
            "recent_closers": [],
        }

    def save_state(self, state: dict) -> None:
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Could not save state: {e}")

    def should_post_today(self, state: dict) -> bool:
        if self.force_post:
            print("FORCE_POST=1 detected. Bypassing once-per-day guard.")
            return True
        today = datetime.now().strftime("%Y-%m-%d")
        if state.get("last_post_date") == today:
            print(f"Already posted today ({today}). Skipping.")
            return False
        return True

    def get_next_topic(self, state: dict) -> Tuple[str, dict]:
        last_topics = state.get("last_topics", [])
        available = [t for t in self.topics if t not in last_topics]
        if not available:
            last_topics = []
            available = list(self.topics.keys())
        next_topic = random.choice(available)
        last_topics.append(next_topic)
        if len(last_topics) > 3:
            last_topics = last_topics[-3:]
        state["last_topics"] = last_topics
        return next_topic, state

    # -------------------------------
    # News helpers (PYMNTS + RSS)
    # -------------------------------
    def fetch_pymnts_news(self, query: str = "", max_items: int = 6) -> List[dict]:
        """
        Scrape recent articles from https://www.pymnts.com/.
        - Scans homepage and a few sections.
        - Filters to pymnts.com links with dated paths or /news/.
        - Scores titles by presence of query terms.
        """
        headers = {"User-Agent": "curl/8"}
        pages = [
            "https://www.pymnts.com/",
            "https://www.pymnts.com/category/news/",
            "https://www.pymnts.com/category/payments/",
            "https://www.pymnts.com/category/b2b/",
        ]
        seen: List[str] = []
        items: List[dict] = []

        def good_url(u: str) -> bool:
            if not u.startswith("http"):
                return False
            if "pymnts.com" not in domain(u):
                return False
            path = urlparse(u).path
            if re.search(r"/\d{4}/\d{2}/\d{2}/", path):  # dated article URL
                return True
            return "/news/" in path

        # collect candidate URLs
        for url in pages:
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code != 200:
                    continue
                for href in re.findall(r'href=["\'](https?://[^"\']+)["\']', r.text, flags=re.IGNORECASE):
                    if good_url(href) and href not in seen:
                        seen.append(href)
            except Exception:
                continue
            if len(seen) > max_items * 4:
                break

        # fetch titles and score
        q_terms = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query)] if query else []
        scored: List[tuple] = []
        for href in seen[: max_items * 6]:
            try:
                rr = requests.get(href, headers=headers, timeout=10)
                if rr.status_code != 200:
                    continue
                title = _extract_meta(rr.text, "og:title") or _extract_meta(rr.text, "twitter:title")
                if not title:
                    continue
                score = 0
                tl = title.lower()
                for t in q_terms:
                    if t and t in tl:
                        score += 1
                scored.append((score, title.strip(), href))
            except Exception:
                continue

        scored.sort(key=lambda x: (-x[0], x[1]))
        for _, title, href in scored[:max_items]:
            items.append({"title": title, "link": href})

        if not items:
            items = [{"title": "Latest PYMNTS coverage on payments and B2B", "link": ""}]
        return items

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

    def fetch_news(self, topic_key: str, query: str, max_items: int = 6) -> List[dict]:
        # Prefer PYMNTS if globally requested, or automatically for the payments topic
        use_pymnts = (self.news_source == "pymnts") or (topic_key == "payments")
        if use_pymnts:
            items = self.fetch_pymnts_news(query=query, max_items=max_items)
            if items:
                return items

        # Fallback: Google News RSS
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items: List[dict] = []
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
            print(f"Warning: no news found for {topic_key}, using fallback.")
            items = [{"title": f"Current industry trends in {topic_key.replace('_', ' ')}", "link": ""}]
        return items

    def pick_publisher_link(self, news_items: List[dict]) -> str:
        """Return a deep publisher link (no Google News, must have a real path)."""
        for it in news_items:
            l = (it.get("link") or "").strip()
            if not l or is_google_news(l):
                continue
            parsed = urlparse(l)
            if parsed.scheme.startswith("http") and parsed.netloc and parsed.path and parsed.path != "/":
                return l
        return ""

    # -------------------------------
    # Prompting & generation
    # -------------------------------
    def _build_prompt(self, topic_key: str, news_items: List[dict], include_link: bool,
                      tone: str, style: str, keep: int = 2) -> str:
        trimmed = []
        for it in news_items[:keep]:
            t = (it["title"] or "").strip()
            title = t[:140] + ("..." if len(t) > 140 else "")
            link = it.get("link") or ""
            link_part = f" | {link}" if link and not is_google_news(link) else ""
            trimmed.append(f"- {title}{link_part}")
        news_context = "\n".join(trimmed)

        link_instruction = (
            "Include exactly one publisher link on its own line before the hashtags. Do not link to news.google.com."
            if include_link and any(x.get("link") and not is_google_news(x["link"]) for x in news_items)
            else "Do not include any links."
        )

        persona = "Write like a senior tech/fintech sales leader who favors partnerships—direct, grounded, helpful."
        structures = {
            "story": "Structure: 1) short scene/opening, 2) what changed, 3) concrete lesson for partnerships/gtm, 4) one-line CTA.",
            "tactical": "Structure: 1) problem, 2) 3 bullet-style tactics in prose, 3) expected outcome, 4) CTA.",
            "contrarian": "Structure: 1) common belief, 2) why it fails, 3) better pattern, 4) example, 5) CTA.",
            "data_point": "Structure: 1) single metric or observation, 2) implication, 3) one play to run, 4) CTA.",
            "playbook": "Structure: 1) context, 2) simple playbook steps, 3) risk to avoid, 4) CTA.",
        }
        tone_line = ("Tone: inspirational with restraint, never chest-beating."
                     if tone == "inspirational" else "Tone: thought leadership, specific and useful.")

        prompt = (
            f"{persona} {tone_line} Topic: {topic_key.replace('_', ' ')}. "
            f"{structures.get(style, 'Structure: 1) context, 2) steps, 3) risk, 4) CTA.')}\n\n"
            "Requirements:\n"
            f"- Strictly {WORD_MIN}–{WORD_MAX} words. Vary sentence length and rhythm.\n"
            "- Avoid buzzwords. No generic openers like 'As a'. No em dashes. No semicolons.\n"
            "- Use a clean, confident, human voice. Keep it practical.\n"
            "- End with exactly 3 relevant hashtags on a final separate line.\n"
            f"- {link_instruction}\n\n"
            f"Recent items to anchor context:\n{news_context}\n\n"
            "Return only the post body with no preface, no headers, no quotes, and no markdown fences."
        )
        return prompt

    def _extract_text_from_gemini(self, payload: dict) -> str | None:
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

    def generate_post_with_gemini(self, topic_key: str, news_items: List[dict],
                                  include_link: bool, tone: str, style: str) -> str | None:
        if not self.gemini_key:
            return None

        # Use models commonly available on free tier
        attempts = [
            {"api": "v1", "model": "gemini-2.5-flash", "keep": 2, "max_out": 520},
        ]
        headers = {"Content-Type": "application/json"}

        for i, step in enumerate(attempts, start=1):
            base = f"https://generativelanguage.googleapis.com/{step['api']}/models/{step['model']}:generateContent"
            prompt = self._build_prompt(topic_key, news_items, include_link, tone, style, keep=step["keep"])
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": step["max_out"]},
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
                    fr = data.get("candidates", [{}])[0].get("finishReason")
                    ttc = data.get("usageMetadata", {}).get("thoughtsTokenCount", 0)
                    if fr == "MAX_TOKENS":
                        bigger = max(step["max_out"], ttc + 300, 1200)
                        body["generationConfig"]["maxOutputTokens"] = bigger
                        print(f"Retrying {step['model']} with maxOutputTokens={bigger} (thinking={ttc})")
                        resp = requests.post(f"{base}?key={self.gemini_key}", headers=headers, json=body, timeout=45)
                        if resp.status_code == 200:
                            text = self._extract_text_from_gemini(resp.json())
                            if text:
                                return self.enforce_style_rules(self.debuzz(text))
                    continue

                return self.enforce_style_rules(self.debuzz(text))
            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue
        return None

    def generate_post_with_openai(self, topic_key: str, news_items: List[dict],
                                  include_link: bool, tone: str, style: str) -> str | None:
        if not self.openai_key:
            return None

        news_context = "\n".join(
            f"- {it['title'][:140]}{'...' if len(it['title'])>140 else ''}"
            f"{' | ' + it['link'] if it.get('link') and not is_google_news(it['link']) else ''}"
            for it in news_items[:2]
        )
        link_instruction = (
            "Include exactly one publisher link on its own line before the hashtags. Do not link to news.google.com."
            if include_link and any(x.get("link") and not is_google_news(x["link"]) for x in news_items)
            else "Do not include any links."
        )
        tone_line = "inspirational" if tone == "inspirational" else "thought leadership"
        structures = {
            "story": "1) opening scene, 2) change, 3) lesson, 4) CTA",
            "tactical": "1) problem, 2) 3 tactics in prose, 3) outcome, 4) CTA",
            "contrarian": "1) common belief, 2) why it fails, 3) better pattern, 4) example, 5) CTA",
            "data_point": "1) metric/observation, 2) implication, 3) play to run, 4) CTA",
            "playbook": "1) context, 2) steps, 3) risk to avoid, 4) CTA",
        }

        prompt = (
            "Write a LinkedIn post from a senior sales leader in tech/fintech who favors partnerships. "
            f"Tone: {tone_line}. Topic: {topic_key.replace('_', ' ')}. Structure: {structures.get(style, 'playbook')}.\n\n"
            f"Requirements:\n- Strictly {WORD_MIN}–{WORD_MAX} words. Vary sentence length and rhythm.\n"
            "- Direct, practical, human. Avoid buzzwords. No generic openers. No em dashes. No semicolons.\n"
            "- End with exactly 3 relevant hashtags on the last line.\n"
            f"- {link_instruction}\n\n"
            f"Recent items:\n{news_context}\n\n"
            "Return only the post body with no preface, no headers, no quotes, and no markdown fences."
        )

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
   python
        body = {
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_completion_tokens": 520,
        }
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

    # -------------------------------
    # Refinement & padding passes
    # -------------------------------
    def expand_if_short(self, text: str, min_words: int = WORD_MIN, target_high: int = WORD_MAX) -> str:
        """If text shorter than min_words, expand into the target band without adding hashtags."""
        if not text or self.word_count(text) >= min_words:
            return text
        prompt = (
            "Expand and refine the LinkedIn post to approximately "
            f"{min_words}-{target_high} words. Keep the same ideas, facts, and any URL. "
            "Add one concrete example or small detail. Avoid hype. "
            "No em dashes. No semicolons. Return only the post body with no preface, no headers, "
            "no quotes, and no markdown fences. End with NO hashtags.\n\nPost:\n" + text
        )
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
                body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 360}}
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"},
                                  json=body, timeout=35)
                if r.status_code == 200:
                    expanded = self._extract_text_from_gemini(r.json())
                    if expanded:
                        return self.enforce_style_rules(self.debuzz(expanded))
            if self.openai_key:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
   ```python
        body = {
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_completion_tokens": 520,
        }
   ```
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                  json=body, timeout=35)
                if r.status_code == 200:
                    expanded = r.json()["choices"][0]["message"]["content"]
                    if expanded:
                        return self.enforce_style_rules(self.debuzz(expanded))
        except Exception:
            pass
        return text

    def _pad_to_minimum(self, text: str, topic_key: str) -> str:
        """Last-mile padding to reach WORD_MIN if a few words short (adds one crisp sentence)."""
        wc = self.word_count(text)
        if wc >= WORD_MIN:
            return text
        extras = {
            "tech_partnerships": "Write the experiment, owner, target metric, and end date where everyone can see it.",
            "ai": "Pilot where data is clean and the decision loop is fast; publish the baseline and lift.",
            "payments": "Track authorization, fraud loss, and time-to-settlement in one place and decide from there.",
            "agentic_commerce": "Constrain agents with budget, catalog, and escalation rules before you scale.",
            "generative_ai": "Tighten the accept-reject loop and fix inputs before you add more prompts.",
            "ai_automations": "Design the stop conditions first; automation without an exit becomes a queue.",
        }
        sentence = extras.get(topic_key, random.choice(INSIGHT_VARIATIONS))
        if not sentence.endswith(('.', '!', '?')):
            sentence += "."
        text = (text.rstrip() + " " + sentence).strip()
        return text

    def trim_to_max_words(self, text: str, max_words: int) -> str:
        if self.word_count(text) <= max_words:
            return text.strip()

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        trimmed = list(sentences)
        while trimmed and self.word_count(" ".join(trimmed)) > max_words:
            trimmed = trimmed[:-1]

        if trimmed:
            candidate = " ".join(trimmed).strip()
            if not self.ends_cleanly(candidate):
                candidate = candidate.rstrip(' "\n') + "."
            return candidate

        words = text.split()
        candidate = " ".join(words[:max_words]).rstrip(', ')
        if not candidate.endswith(('.', '!', '?')):
            candidate = candidate + '.'
        return candidate.strip()

    def polish_with_model(self, text: str) -> str:
        if not text:
            return text
        prompt = (
            "Edit the LinkedIn post to improve grammar, flow, and clarity. "
            "Keep the same ideas and any URL. Remove any hashtags you see. "
            "Neutral, senior voice. No em dashes. No semicolons. "
            "Return only the post body with no preface, no headers, no quotes, and no markdown fences.\n\n"
            f"Post:\n{text}"
        )
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
                body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 320}}
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"},
                                  json=body, timeout=35)
                if r.status_code == 200:
                    edited = self._extract_text_from_gemini(r.json())
                    if edited:
                        return self.enforce_style_rules(self.debuzz(edited))
            if self.openai_key:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
   ```python
        body = {
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_completion_tokens": 520,
        }
   ```
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                  json=body, timeout=35)
                if r.status_code == 200:
                    edited = r.json()["choices"][0]["message"]["content"]
                    if edited:
                        return self.enforce_style_rules(self.debuzz(edited))
        except Exception as e:
            print(f"Polish pass skipped due to error: {e}")
        return self.enforce_style_rules(self.debuzz(text))

    # -------------------------------
    # Hashtags & finalization helpers
    # -------------------------------
    def _keywords_from_titles(self, news_items: List[dict], k: int = 6) -> List[str]:
        text = " ".join((it.get("title") or "") for it in news_items[:4])
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
        freq: Dict[str, int] = {}
        for w in words:
            wl = w.lower()
            if wl in self.STOPWORDS or len(wl) < 4:
                continue
            freq[wl] = freq.get(wl, 0) + 1
        keys = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]

        def camel(t: str) -> str:
            return re.sub(r"[^A-Za-z0-9]", "", t.title())

        return [camel(t[0]) for t in keys if t[0]]

    def curated_hashtags(self, topic_key: str, body: str,
                         news_items: List[dict], state: dict) -> str:
        base = self.topic_tags.get(topic_key, ["#B2B", "#GTM"])
        dynamic = ["#" + k for k in self._keywords_from_titles(news_items, k=8) if k]
        pool = list(dict.fromkeys(base + dynamic))  # unique, keep order

        body_lower = body.lower()

        def is_ok(tag: str) -> bool:
            tl = tag.lower()
            if tl in self.BANNED_BRAND_TAGS and tl[1:] not in body_lower:
                return False
            return True

        candidates = [t for t in pool if is_ok(t) and t.lower() not in body_lower]

        recent_sets = state.get("recent_hashtag_sets", [])
        random.shuffle(candidates)
        chosen = None
        for i in range(min(20, len(candidates))):
            trio = candidates[i:i + 3]
            if len(trio) < 3:
                break
            trial = " ".join(trio)
            if trial and trial.lower() not in recent_sets:
                chosen = trio
                break

        if not chosen:
            fallback = [t for t in base if t.lower() not in self.BANNED_BRAND_TAGS]
            while len(fallback) < 3:
                for g in ["#B2B", "#GTM", "#AI", "#EnterpriseAI"]:
                    if g not in fallback:
                        fallback.append(g)
                    if len(fallback) == 3:
                        break
            chosen = fallback[:3]

        result = " ".join(chosen)
        recent_sets.append(result.lower())
        state["recent_hashtag_sets"] = recent_sets[-10:]
        return result

    def inject_open_close(self, body: str, state: dict) -> Tuple[str, str]:
        op = random.choice(self.openers)
        # rotate closers avoiding recent repeats
        recent_closers = state.get("recent_closers", [])
        random.shuffle(self.closers)
        closer = None
        for c in self.closers:
            if c.lower() not in recent_closers:
                closer = c
                break
        closer = closer or random.choice(self.closers)
        recent_closers.append(closer.lower())
        state["recent_closers"] = recent_closers[-10:]

        body = body.strip()
        new_body = f"{op} {body}"
        return new_body, closer

    def dedupe_sentences(self, text: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        seen = set()
        clean = []
        for s in parts:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                clean.append(s.strip())
        return " ".join(clean)

    def _strip_repeated_playbook(self, text: str) -> str:
        if not text:
            return text
        patterns = [
            r"Start with one use case.*?(weekly|week)\.",
            r"Focus on one measurable outcome.*?(repeatable|scale)\.",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s{2,}", " ", text).strip()

    def sanitize_and_finalize(self, body: str, topic_key: str, include_link: bool,
                              news_items: List[dict], state: dict) -> str:
        # 1) strip model wrappers first
        body = self.strip_model_wrappers(body)

        # Normalize 'hashtag#Tag' artifacts to '#Tag' then remove stray inline hashtags
        body = re.sub(r"\bhashtag#", "#", body, flags=re.IGNORECASE)

        # remove any hashtag lines the model added, anywhere
        lines = [ln for ln in body.splitlines() if not re.fullmatch(r'\s*(#\w+\s*){2,}\s*', ln.strip(), flags=re.IGNORECASE)]
        body = "\n".join(lines)

        # remove inline stray hashtags completely — we always rebuild them
        body = re.sub(r'(?<!\w)#\w+', "", body)

        body = self.enforce_style_rules(self.debuzz(body)).strip()

        # ensure no trailing hashtag lines survived
        m = re.search(r'(^|\n)\s*(#\w+(?:\s+#\w+){1,})\s*$', body, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            body = body[: m.start()].rstrip()

        # Expansion if short
        body = self.expand_if_short(body, min_words=WORD_MIN, target_high=WORD_MAX)

        # De-dupe and ensure clean end
        body = self.dedupe_sentences(body)
        if not self.ends_cleanly(body):
            body = body.rstrip(' "\n') + "."

        # Inject opener and closer
        body, closer = self.inject_open_close(body, state)
        if not self.ends_cleanly(body):
            body += " "
        body += closer
        if not self.ends_cleanly(body):
            body += "."

        # Last-mile pad if still short and trim if needed
        body = self._pad_to_minimum(body, topic_key)
        if self.word_count(body) > WORD_MAX:
            body = self.trim_to_max_words(body, WORD_MAX)
        if self.word_count(body) < WORD_MIN:
            body = self._pad_to_minimum(body, topic_key)

        # Link (deep publisher only)
        link_line = ""
        if include_link:
            good = self.pick_publisher_link(news_items)
            if not good and os.getenv("REQUIRE_LINK") == "1":
                # Try a different query once to find a publisher link
                alt_query = random.choice(self.topics.get(topic_key, []))
                print(f"No clean publisher link found; retrying news fetch for deep link: {alt_query}")
                alt_items = self.fetch_news(topic_key, alt_query)
                good = self.pick_publisher_link(alt_items)
                if not good:
                    print("Still no clean publisher link after retry.")
            if good:
                link_line = f"\n\n{good}"

        # Hashtags — ALWAYS rebuild at the end
        tags = self.curated_hashtags(topic_key, body, news_items, state)
        hashtags_line = f"\n\n{tags}" if tags else ""

        final_text = body + link_line + hashtags_line
        return self.enforce_style_rules(self.debuzz(final_text)).strip()

    def quality_gate(self, text: str, include_link: bool) -> Tuple[bool, str]:
        """Return (ok, reason) to decide if a post is publishable."""
        if not text:
            return False, "empty"

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return False, "empty"

        hashtag_line = ""
        if lines and HASHTAG_BLOCK_RE.match(lines[-1]):
            hashtag_line = lines.pop()
            tags = hashtag_line.split()
            if len(tags) != 3:
                return False, "invalid hashtag count"
        elif any("#" in l for l in lines):
            return False, "hashtags not at end as block"

        link_line = ""
        if lines and re.match(r"^https?://", lines[-1], flags=re.IGNORECASE):
            link_line = lines.pop()

        body_text = " ".join(lines).strip()
        if not body_text:
            return False, "empty body"

        wc = self.word_count(body_text)
        if wc < WORD_MIN:
            return False, f"too short ({wc} words)"
        if wc > WORD_MAX:
            return False, f"too long ({wc} words)"
        if len(re.findall(r"[.!?]", body_text)) < 3:
            return False, "too few sentences"

        if include_link:
            if not link_line:
                return False, "missing publisher link"
            if "news.google.com" in link_line.lower():
                return False, "google news link present"
        elif link_line:
            return False, "unexpected link line"

        return True, "ok"

    # -------------------------------
    # LinkedIn posting
    # -------------------------------
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
                print("Posted to LinkedIn")
                return True
            print(f"Post failed: {r.status_code} - {r.text[:400]}")
            if r.status_code == 401:
                print("Hint: token expired or missing w_member_social scope.")
            if r.status_code == 403:
                print("Hint: app lacks member post permission.")
            return False
        except Exception as e:
            print(f"LinkedIn error: {e}")
            return False

    # -------------------------------
    # Main run
    # -------------------------------
    def run_weekly_post(self) -> None:
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        if not (self.gemini_key or self.openai_key):
            return
        if not (self.linkedin_token and self.person_urn):
            return

        state = self.load_state()
        if not self.should_post_today(state):
            return

        # Randomize time in window (skipped if SKIP_DELAY/DISABLE_DELAY=1)
        random_delay_minutes(min_minutes=0, max_minutes=120)

        topic_key, state = self.get_next_topic(state)
        include_link = random.random() < 0.6
        if os.getenv("REQUIRE_LINK") == "1":
            include_link = True
        tone = random.choice(["inspirational", "thought_leadership"])
        style = random.choice(self.style_modes)

        print(f"Topic: {topic_key} | tone: {tone} | style: {style} | link: {include_link}")
        query = random.choice(self.topics[topic_key])
        print(f"Fetching news for: {query}")
        news_items = self.fetch_news(topic_key, query)
        print(f"Found {len(news_items)} items")

        # Link policy
        if include_link and not any(it.get("link") and not is_google_news(it["link"]) for it in news_items):
            # If REQUIRE_LINK=1 we'll retry later inside sanitize_and_finalize
            if os.getenv("REQUIRE_LINK") != "1":
                include_link = False
            print("No clean publisher link found, switching to linkless piece." if os.getenv("REQUIRE_LINK") != "1"
                  else "No clean publisher link found; will retry once for a deep link during finalize.")

        # Compose
        post_body = None
        if self.gemini_key:
            print("Generating with Gemini")
            post_body = self.generate_post_with_gemini(topic_key, news_items, include_link, tone, style)
        if not post_body and self.openai_key:
            print("Fallback to OpenAI")
            post_body = self.generate_post_with_openai(topic_key, news_items, include_link, tone, style)
        if not post_body:
            raise RuntimeError("Unable to generate post with available models.")

        # Clean/shape
        post_body = self._strip_repeated_playbook(post_body)
        post_body = self.polish_with_model(post_body)
        final_text = self.sanitize_and_finalize(post_body, topic_key, include_link, news_items, state)

        # Gate quality
        ok_q, reason = self.quality_gate(final_text, include_link)
        if not ok_q:
            raise RuntimeError(f"Quality gate failed: {reason}")

        print("\nGenerated post\n" + "-" * 60 + f"\n{final_text}\n" + "-" * 60)

        ok = self.post_to_linkedin(final_text)
        if ok:
            state["post_count"] = state.get("post_count", 0) + 1
            state["last_post_date"] = datetime.now().strftime("%Y-%m-%d")
            self.save_state(state)
            print(f"Success. Total posts: {state['post_count']}")
        else:
            raise RuntimeError("Publish failed")


if __name__ == "__main__":
    LinkedInAIAgent().run_weekly_post()
