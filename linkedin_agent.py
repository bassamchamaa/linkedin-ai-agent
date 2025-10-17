#!/usr/bin/env python3
"""
LinkedIn AI Agent — stable length, clean hashtags, no instruction leakage.

Key features:
- Strict word window with measured retries (MIN_WORDS..MAX_WORDS).
- Expands body first (no hashtags), then regenerates hashtags for final line.
- Strips inline hashtags (#Tag and 'hashtag#Tag'), blocks instruction/meta leakage.
- Deep-link resolver avoids Google News; optional REQUIRE_LINK / ABORT_IF_NO_LINK.
- No QC bypass on FORCE_POST; FORCE_POST only skips once-per-day guard.
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
    """Sleep for a random amount of time unless SKIP_DELAY/DISABLE_DELAY is set."""
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

# --------------------------------
# Constants for style/variation
# --------------------------------
INSIGHT_VARIATIONS = [
    "Pick one use case, instrument it, and publish real numbers weekly.",
    "Anchor on one measurable outcome and review progress in the open.",
    "Make the play repeatable: same metric, same cadence, broader surface.",
    "Ship something small, measure honestly, scale what actually works.",
]

# -------------------------------
# Main Agent
# -------------------------------
class LinkedInAIAgent:
    # Hashtag stopwords + brand bans
    STOPWORDS = {
        "the","and","for","with","that","this","from","into","over","after",
        "about","your","their","there","of","a","to","in","on","at","as","by",
        "is","are","was","be","it","its","an","or","we","our","you","they",
        "ai","llm","llms","via","vs","using","how","why","what","new","latest",
        "future","today","more"
    }
    BANNED_BRAND_TAGS = {"#paypal", "#visa", "#mastercard", "#stripe", "#adyen"}

    # Length controls (BODY ONLY; hashtags excluded)
    MIN_WORDS = 150
    MAX_WORDS = 210
    TARGET_HIGH = 190  # internal expansion target

    # Content-safety guard: if any of these appear, we refuse to post
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
    # News helpers
    # -------------------------------
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
        """Return a deep publisher link by resolving redirects if needed (no Google News, must have real path)."""
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

    # -------------------------------
    # Prompting & generation
    # -------------------------------
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
        """Remove inline hashtags including 'hashtag#Word' variants and block hashtag-only lines."""
        lines = [ln for ln in text.splitlines() if not re.fullmatch(r'\s*(#\w+\s*){2,}\s*', ln.strip())]
        text = "\n".join(lines)
        text = re.sub(r'\bhashtag#\w+\b', "", text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\w)#\w+', "", text)
        return re.sub(r'\s{2,}', ' ', text).strip()

    def generate_post_with_gemini(self, topic_key, news_items, include_link, tone, style) -> Optional[str]:
        if not self.gemini_key:
            return None

        def call_gemini(prompt: str, max_out: int) -> Optional[str]:
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.6,
                    "maxOutputTokens": max_out,
                    "responseMimeType": "text/plain"
                },
            }
            try:
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"},
                                  json=body, timeout=45)
                if r.status_code != 200:
                    print(f"Gemini error {r.status_code}: {r.text[:400]}")
                    return None
                data = r.json()
                text = self._extract_text_from_gemini(data)
                return text
            except Exception as e:
                print(f"Gemini call failed: {e}")
                return None

        base_prompt = self._build_prompt(topic_key, news_items, include_link, tone, style, keep=3)

        attempt_prompts = [base_prompt]
        max_tokens_seq = [900, 1100, 1300]

        last_clean = None
        for i in range(3):
            print(f"Gemini bounded attempt {i+1} (maxTokens={max_tokens_seq[i]})")
            raw = call_gemini(attempt_prompts[i], max_tokens_seq[i])
            if not raw:
                continue

            # scrub & count (no hashtags yet)
            clean = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(raw)))
            wc = self.word_count(clean)
            print(f"Model draft words: {wc}")

            if wc == 0:
                continue
            last_clean = clean

            if self.MIN_WORDS <= wc <= self.MAX_WORDS:
                return clean

            # corrective follow-up
            if wc < self.MIN_WORDS:
                delta = self.MIN_WORDS - wc
                corrective = (
                    f"You wrote {wc} words (short by {delta}). Expand the SAME post to "
                    f"{self.MIN_WORDS}-{self.MAX_WORDS} words by adding one concrete example and one outcome. "
                    "Return ONLY the revised post text. Keep the exact formatting rules (final line is exactly 3 hashtags)."
                )
            else:
                corrective = (
                    f"You wrote {wc} words (over the limit). Trim to {self.MIN_WORDS}-{self.MAX_WORDS} words "
                    "by removing filler while preserving the core idea. Return ONLY the revised post. "
                    "Keep the exact formatting rules (final line is exactly 3 hashtags)."
                )
            attempt_prompts.append(clean + "\n\n" + corrective)

        # If still out of range, try local boost to MIN_WORDS
        if last_clean:
            boosted = self.expand_body_if_short(last_clean)
            boosted = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(boosted)))
            if self.word_count(boosted) >= self.MIN_WORDS:
                return boosted

        return last_clean

    def generate_post_with_openai(self, topic_key: str, news_items: List[dict],
                                  include_link: bool, tone: str, style: str) -> Optional[str]:
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
            "STRICT OUTPUT RULES (must follow ALL):\n"
            f"- Write between {self.MIN_WORDS} and {self.MAX_WORDS} words inclusive. Count your words.\n"
            "- No bullets or numbering in the output. Use paragraphs only.\n"
            "- Avoid buzzwords. No generic openers. No em dashes. No semicolons.\n"
            "- End with exactly 3 relevant hashtags on the last line (single line, 3 tags).\n"
            f"- {link_instruction}\n"
            "- Return ONLY the post. Do not add headings, notes, or explanations.\n\n"
            f"Recent items:\n{news_context}\n"
        )

        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6, "max_tokens": 900}
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=45)
            if r.status_code != 200:
                print(f"OpenAI error {r.status_code}: {r.text[:400]}")
                return None
            text = r.json()["choices"][0]["message"]["content"]
            return self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(text)))
        except Exception as e:
            print(f"OpenAI call failed: {e}")
            return None

    # -------------------------------
    # Refinement passes
    # -------------------------------
    def expand_body_if_short(self, body: str) -> str:
        """Expand BODY ONLY (no hashtags inside). Try model; then guarantee locally."""
        if not body or self.word_count(body) >= self.MIN_WORDS:
            return body

        prompt = (
            "Expand and refine this LinkedIn post body to approximately "
            f"{self.MIN_WORDS}-{self.TARGET_HIGH} words. Keep the same ideas and any URL. "
            "Add one concrete example or small detail. Avoid hype. "
            "No em dashes. No semicolons. Do NOT add hashtags.\n\n"
            f"Body:\n{body}"
        )
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
                body_req = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 520}}
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"},
                                  json=body_req, timeout=35)
                if r.status_code == 200:
                    expanded = self._extract_text_from_gemini(r.json())
                    if expanded:
                        body = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(expanded)))
            elif self.openai_key:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                body_req = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3, "max_tokens": 520}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                  json=body_req, timeout=35)
                if r.status_code == 200:
                    expanded = r.json()["choices"][0]["message"]["content"]
                    if expanded:
                        body = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(expanded)))
        except Exception:
            pass

        # Pure-Python pad if still short
        if self.word_count(body) < self.MIN_WORDS:
            pads = [
                " Outline the owner, the metric, and the review cadence so decisions are fast and visible.",
                " Share one recent example with a baseline, the change you observed, and the next step so progress compounds.",
                " Remove one handoff that creates rework. Add one alert that prevents silent failure in production.",
                " Write down the ‘stop conditions’ so automation exits cleanly and humans re-enter with context.",
                " Close the loop by publishing results weekly. Make it boring, repeatable, and easy to copy.",
            ]
            i = 0
            while self.word_count(body) < self.MIN_WORDS and i < len(pads):
                if not self.ends_cleanly(body):
                    body = body.rstrip() + "."
                body += pads[i]
                i += 1
            if self.word_count(body) < self.MIN_WORDS:
                body += " For example, pick one flow, set a baseline, and publish a two-week improvement log."
        return body

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

    def assemble_post(self, raw_body: str, topic_key: str, include_link: bool,
                      news_items: List[dict], state: dict) -> str:
        """
        Pipeline:
        1) Strip any hashtags in body (including 'hashtag#').
        2) Clean style + debuzz.
        3) Dedupe sentences + ensure terminal punctuation.
        4) Inject opener/closer.
        5) Attach deep link (if allowed & found) — no hashtags yet.
        6) Expand BODY (+link) to MIN_WORDS.
        7) Build hashtags from expanded body; append as final single line.
        """
        body = self._strip_hashtags_anywhere(raw_body)
        body = self.enforce_style_rules(self.debuzz(body)).strip()

        body = self.dedupe_sentences(body)
        if not self.ends_cleanly(body):
            body = body.rstrip(' "\n') + "."

        body, closer = self.inject_open_close(body, state)
        if not self.ends_cleanly(body):
            body += " "
        body += closer
        if not self.ends_cleanly(body):
            body += "."

        link_line = ""
        if include_link:
            good = self.pick_publisher_link(news_items)
            if good:
                link_line = f"\n\n{good}"
            elif os.getenv("ABORT_IF_NO_LINK") == "1":
                raise RuntimeError("ABORT_IF_NO_LINK=1 and no deep link found.")
        body_plus_link = body + link_line

        expanded = self.expand_body_if_short(body_plus_link)

        tags = self.curated_hashtags(topic_key, expanded, news_items, state)
        final_text = expanded + (f"\n\n{tags}" if tags else "")

        return self.enforce_style_rules(self.debuzz(final_text)).strip()

    # -------------------------------
    # Quality gate
    # -------------------------------
    def has_meta_leakage(self, text: str) -> bool:
        t = text or ""
        return any(flag.lower() in t.lower() for flag in self.META_RED_FLAGS)

    def quality_gate(self, text: str, include_link: bool) -> Tuple[bool, str]:
        if not text:
            return False, "empty"

        if self.has_meta_leakage(text):
            return False, "meta/instructions leakage detected"

        lines = [l for l in text.splitlines() if l.strip()]
        hashtags_line = ""
        if lines and re.fullmatch(r"(#\w+\s*){3}", lines[-1].strip()):
            hashtags_line = lines[-1]
        elif lines and re.fullmatch(r"(#\w+\s*){2,}", lines[-1].strip()):
            # If not exactly 3, fail
            return False, "hashtags count not equal to 3 on last line"

        body_only = text
        if hashtags_line:
            body_only = text[: text.rfind(hashtags_line)].rstrip()

        wc = self.word_count(body_only)
        if wc < self.MIN_WORDS:
            return False, f"too short ({wc} words)"
        if wc > self.MAX_WORDS:
            return False, f"too long ({wc} words)"
        if len(re.findall(r"[.!?]", body_only)) < 3:
            return False, "too few sentences"

        if "#" in text and not hashtags_line:
            return False, "hashtags not at end as single line"

        if include_link and "news.google.com" in text:
            return False, "google news link present"

        # no lines that are just hashtags mid-body
        mid = [ln for ln in lines[:-1]]
        for ln in mid:
            if re.fullmatch(r"(#\w+\s*){2,}", ln):
                return False, "inline hashtag block found"

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
    # Local compose fallback
    # -------------------------------
    def local_compose(self, topic_key: str) -> str:
        hooks = {
            "tech_partnerships": "Partnerships work when both teams invest real resources and chase one metric together.",
            "ai": "AI wins in the enterprise when it reduces time to value, not just cost.",
            "payments": "Payments is a trust business. Speed and acceptance matter, proof beats promises.",
            "agentic_commerce": "Agentic commerce turns browsing into doing. The best agents remove steps, not just clicks.",
            "generative_ai": "Generative AI helps when it is paired with data quality and strong guardrails.",
            "ai_automations": "Automation shines when it owns the boring work and hands off the tough parts clearly.",
        }
        hook = hooks.get(topic_key, "Clarity beats noise.")
        insight = random.choice(INSIGHT_VARIATIONS)
        return self.enforce_style_rules(self.debuzz(f"{hook} {insight}"))

    # -------------------------------
    # Main run
    # -------------------------------
    def run_weekly_post(self) -> None:
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        state = self.load_state()
        if not self.should_post_today(state):
            return

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

        if include_link and not any(it.get("link") and not is_google_news(it["link"]) for it in news_items):
            if os.getenv("REQUIRE_LINK") == "1":
                print("REQUIRE_LINK=1 and no clean publisher link found. Aborting this run.")
                return
            include_link = False
            print("No clean publisher link found, switching to linkless piece.")

        try:
            post_body = None
            if self.gemini_key:
                print("Generating with Gemini")
                post_body = self.generate_post_with_gemini(topic_key, news_items, include_link, tone, style)
            if not post_body and self.openai_key:
                print("Fallback to OpenAI")
                post_body = self.generate_post_with_openai(topic_key, news_items, include_link, tone, style)
            if not post_body:
                print("Models failed, composing locally")
                post_body = self.local_compose(topic_key)

            post_body = self._strip_repeated_playbook(post_body)
            post_body = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(post_body)))

            final_text = self.assemble_post(post_body, topic_key, include_link, news_items, state)

            ok_q, reason = self.quality_gate(final_text, include_link)
            if not ok_q:
                print(f"Quality gate failed: {reason}. Falling back to local compose and re-assemble.")
                fallback = self.local_compose(topic_key)
                fallback = self.enforce_style_rules(self.debuzz(self._strip_hashtags_anywhere(fallback)))
                final_text = self.assemble_post(fallback, topic_key, include_link=False, news_items=news_items, state=state)

                ok_q2, reason2 = self.quality_gate(final_text, include_link=False)
                if not ok_q2:
                    print(f"Final quality gate failed: {reason2}. Not posting.")
                    return

            print("\nGenerated post\n" + "-" * 60 + f"\n{final_text}\n" + "-" * 60)

            # Final meta leak check before posting
            if self.has_meta_leakage(final_text):
                print("Aborting: instruction/meta leakage detected.")
                return

            ok = self.post_to_linkedin(final_text)
            if ok:
                state["post_count"] = state.get("post_count", 0) + 1
                state["last_post_date"] = datetime.now().strftime("%Y-%m-%d")
                self.save_state(state)
                print(f"Success. Total posts: {state['post_count']}")
            else:
                print("Publish failed")
        except RuntimeError as e:
            print(str(e))
            return


if __name__ == "__main__":
    LinkedInAIAgent().run_weekly_post()
