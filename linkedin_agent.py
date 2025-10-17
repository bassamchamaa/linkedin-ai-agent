#!/usr/bin/env python3
"""
LinkedIn AI Agent — stable length, clean hashtags, no instruction leakage.

Fixes in this build:
- Remove unsupported `responseMimeType` from Gemini requests (avoids HTTP 400).
- Robust Gemini retry loop (no IndexError if a prior attempt fails to return text).
- Retains strict word window, body-first expansion, final-line hashtags, deep-linking.
"""

import os
import json
import re
import random
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Optional

import requests

# -------------------------------
# Utility: randomized post delay
# -------------------------------
def random_delay_minutes(min_minutes: int = 0, max_minutes: int = 120) -> None:
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
    return d.endswith("news.google.com") or (d.endswith("google.com") and "/articles/" in url)

INSIGHT_VARIATIONS = [
    "Pick one use case, instrument it, and publish real numbers weekly.",
    "Anchor on one measurable outcome and review progress in the open.",
    "Make the play repeatable: same metric, same cadence, broader surface.",
    "Ship something small, measure honestly, scale what actually works.",
]

class LinkedInAIAgent:
    STOPWORDS = {
        "the","and","for","with","that","this","from","into","over","after",
        "about","your","their","there","of","a","to","in","on","at","as","by",
        "is","are","was","be","it","its","an","or","we","our","you","they",
        "ai","llm","llms","via","vs","using","how","why","what","new","latest",
        "future","today","more"
    }
    BANNED_BRAND_TAGS = {"#paypal", "#visa", "#mastercard", "#stripe", "#adyen"}

    MIN_WORDS = 150
    MAX_WORDS = 210
    TARGET_HIGH = 190

    META_RED_FLAGS = [
        "STRICT OUTPUT RULES", "Return ONLY the post", "Do not add headings",
        "You wrote", "short by", "over the limit", "Expand and refine this",
        "Keep the exact formatting rules", "responseMimeType", "maxOutputTokens"
    ]

    def __init__(self) -> None:
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.force_post = os.getenv("FORCE_POST", "").strip() == "1"

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
                "autonomous workflow enterprise adoption",
                "intelligent automation ROI enterprise",
                "AI workforce augmentation impact",
                "AI automation future of work",
                "autonomous business operations future",
                "AI workforce collaboration future",
                "intelligent automation enterprise trends",
                "AI process optimization future",
            ],
        }

        self.topic_tags: Dict[str, List[str]] = {
            "tech_partnerships": ["#Partnerships", "#Ecosystem", "#GTM", "#B2B", "#SaaS"],
            "ai": ["#AI", "#EnterpriseAI", "#B2B", "#GTM", "#ML"],
            "payments": ["#Fintech", "#Payments", "#EmbeddedFinance", "#Risk", "#B2B"],
            "agentic_commerce": ["#AgenticAI", "#Ecommerce", "#CX", "#Conversion"],
            "generative_ai": ["#GenerativeAI", "#LLMs", "#EnterpriseAI", "#Product"],
            "ai_automations": ["#Automation", "#AIAgents", "#Ops", "#Productivity"],
        }

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

        self.state_file = "agent_state.json"

        if not self.linkedin_token or not self.person_urn:
            print("Missing LinkedIn secrets. Set LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN.")
        if not self.gemini_key and not self.openai_key:
            print("No model key found. Set GEMINI_API_KEY or OPENAI_API_KEY.")

    # ---------- style helpers ----------
    def enforce_style_rules(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("—", ",").replace("–", ",").replace(";", ",")
        text = re.sub(r"^\s*As a [^.!\n]+[, ]+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\s*!\s*", "! ", text)
        text = re.sub(r"\s*\?\s*", "? ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([,!.?])", r"\1", text)
        if text.count('"') % 2 == 1 and text.strip().startswith('"'):
            text = text.lstrip('"').lstrip()
        return text.strip()

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

    # ---------- persistence ----------
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

    # ---------- news ----------
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

    def fetch_news(self, topic_key: str, query: str, max_items: int = 12) -> List[dict]:
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
        for it in news_items:
            raw = (it.get("link") or "").strip()
            if not raw:
                continue
            if not is_google_news(raw):
                parsed = urlparse(raw)
                if parsed.scheme.startswith("http") and parsed.netloc and parsed.path and parsed.path != "/":
                    return raw
            try:
                h = {"User-Agent": "curl/8"}
                r = requests.head(raw, allow_redirects=True, timeout=10, headers=h)
                final_url = r.url
                if is_google_news(final_url):
                    r = requests.get(raw, allow_redirects=True, timeout=12, headers=h)
                    final_url = r.url
                if final_url and not is_google_news(final_url):
                    p = urlparse(final_url)
                    if p.scheme.startswith("http") and p.netloc and p.path and p.path != "/":
                        return final_url
            except Exception:
                continue
        return ""

    # ---------- prompting ----------
    def _build_prompt(self, topic_key: str, news_items: List[dict], include_link: bool,
                      tone: str, style: str, keep: int = 3) -> str:
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
            "story": "1) short scene/opening, 2) what changed, 3) concrete lesson for partnerships/gtm, 4) one-line CTA.",
            "tactical": "1) problem, 2) 3 bullet-style tactics in prose, 3) expected outcome, 4) CTA.",
            "contrarian": "1) common belief, 2) why it fails, 3) better pattern, 4) example, 5) CTA.",
            "data_point": "1) single metric or observation, 2) implication, 3) one play to run, 4) CTA.",
            "playbook": "1) context, 2) steps, 3) risk to avoid, 4) CTA.",
        }
        structure = structures.get(style, structures["playbook"])
        tone_line = ("Tone: inspirational with restraint, never chest-beating."
                     if tone == "inspirational" else "Tone: thought leadership, specific and useful.")

        return (
            f"{persona} {tone_line} Topic: {topic_key.replace('_',' ')}. Structure: {structure}\n\n"
            "STRICT OUTPUT RULES (must follow ALL):\n"
            f"- Write between {self.MIN_WORDS} and {self.MAX_WORDS} words inclusive. Count your words.\n"
            "- No bullets or numbering in the output. Use paragraphs only.\n"
            "- Avoid buzzwords. No generic openers like 'As a'. No em dashes. No semicolons.\n"
            "- Use a clean, confident, human voice. Keep it practical.\n"
            "- End with exactly 3 relevant hashtags on the final line (single line with 3 space-separated hashtags).\n"
            f"- {link_instruction}\n"
            "- Return ONLY the post. Do not add headings, notes, or explanations.\n\n"
            f"Recent items to anchor context:\n{news_context}\n"
        )

    def _extract_text_from_gemini(self, payload: dict) -> Optional[str]:
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

    def _strip_hashtags_anywhere(self, text: str) -> str:
        li
