#!/usr/bin/env python3
"""
LinkedIn AI Agent — 150–200 words, resilient deep-linking

Fixes:
- Enforce 150–200 words end-to-end (prompt, expansion, gate + final pad).
- Retry news once with a different query to find a deep publisher link.
- Prefer reputable publisher allow-list when picking links.
- Use available Gemini models: 2.5-flash-lite first (free), then 2.5-flash with MAX_TOKENS-aware retry.
- Normalizes weak hashtags (e.g., "#Agent" -> "#AIAgents").

Env flags:
- SKIP_DELAY=1 or DISABLE_DELAY=1  → skip random wait
- FORCE_POST=1                     → bypass “once per day”
- REQUIRE_LINK=1                   → keep trying to include a real publisher link
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

WORD_MIN = 150
WORD_MAX = 200

ALLOWED_PUBLISHERS = {
    # prefer these when present
    "techcrunch.com", "theverge.com", "wsj.com", "bloomberg.com", "reuters.com",
    "ft.com", "forbes.com", "axios.com", "protocol.com", "cnbc.com",
    "venturebeat.com", "wired.com", "theinformation.com", "seekingalpha.com",
    "paymentsdive.com", "finextra.com", "pymnts.com"
}

def random_delay_minutes(min_minutes: int = 0, max_minutes: int = 120) -> None:
    if os.getenv("SKIP_DELAY") == "1" or os.getenv("DISABLE_DELAY") == "1":
        print("Delay disabled (SKIP_DELAY/DISABLE_DELAY set) — skipping randomized wait.")
        return
    delay_seconds = random.randint(min_minutes * 60, max_minutes * 60)
    print(f"Random delay: waiting {delay_seconds/60:.1f} minutes before posting...")
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
    "Make the metric visible to both teams and review it weekly. Public scoreboards change behavior.",
    "Reduce the first value moment. Shorter time-to-value beats feature depth when budgets are tight.",
    "Instrument the handoffs. Most friction hides between systems, not within them.",
    "Prove the pattern in one segment, then clone it with the same playbook and safeguards.",
]

class LinkedInAIAgent:
    STOPWORDS = {"the","and","for","with","that","this","from","into","over","after",
                 "about","your","their","there","of","a","to","in","on","at","as","by",
                 "is","are","was","be","it","its","an","or","we","our","you","they",
                 "ai","llm","llms","via","vs","using","how","why","what","new","latest",
                 "future","today","more"}
    BANNED_BRAND_TAGS = {"#paypal", "#visa", "#mastercard", "#stripe", "#adyen"}

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
                "Oracle AI integration partnerships",
                "Replit enterprise developer tools",
                "Cursor AI developer productivity",
                "AI agents enterprise productivity",
            ],
            "payments": [
                "Stripe strategic partnerships fintech",
                "Visa payment ecosystem partnerships",
                "Mastercard fintech alliance deals",
                "Adyen merchant partnerships",
                "PayPal business partnership strategy",
                "Block Square merchant partnerships",
                "Fiserv payment partnerships",
                "Checkout.com enterprise payments",
                "Worldpay enterprise merchant services",
                "FIS payment technology enterprise",
                "Plaid financial data connectivity",
                "Wise cross-border payments enterprise",
                "real-time payments business impact",
            ],
            "agentic_commerce": [
                "AI shopping agents ecommerce launch",
                "conversational commerce AI enterprise",
                "Shopify AI merchant partnerships",
                "AI agents ecommerce transformation",
            ],
            "generative_ai": [
                "OpenAI GPT enterprise launch",
                "Microsoft Copilot enterprise update",
                "Google Gemini enterprise features",
                "LLM enterprise automation future",
            ],
            "ai_automations": [
                "AI workflow automation enterprise launch",
                "AI agent orchestration platform",
                "AI agents business process transformation",
                "intelligent automation ROI enterprise",
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
        if not text: return text
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
        if not text: return text
        repl = {r"\bsynergy\b":"fit", r"\bleverage\b":"use", r"\bparadigm\b":"approach",
                r"\bgame[- ]changer\b":"big step", r"\bcutting edge\b":"leading",
                r"\bdisrupt(ive|ion)\b":"shift"}
        for p, r in repl.items():
            text = re.sub(p, r, text, flags=re.IGNORECASE)
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
        return {"last_topics": [], "post_count": 0, "last_post_date": None,
                "recent_hashtag_sets": [], "recent_closers": []}

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
        available = [t for t in self.topics if t not in last_topics] or list(self.topics.keys())
        next_topic = random.choice(available)
        last_topics.append(next_topic)
        state["last_topics"] = last_topics[-3:]
        return next_topic, state

    # ---------- news helpers ----------
    def _extract_original_from_link(self, link: str) -> str:
        try:
            if not link: return ""
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
        if not description: return ""
        try:
            urls = re.findall(r'href="(https?://[^"]+)"', description)
            for u in urls:
                if not is_google_news(u):
                    return u
        except Exception:
            pass
        return ""

    def fetch_news(self, topic_key: str, query: str, max_items: int = 6) -> List[dict]:
        rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        items: List[dict] = []
        try:
            r = requests.get(rss, timeout=12, headers={"User-Agent": "curl/8"})
            if r.status_code != 200:
                print(f"News fetch error {r.status_code} for {topic_key}: {r.text[:200]}")
                return items
            root = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None: return items
            for it in channel.findall("item")[: max_items * 2]:
                title = (it.findtext("title") or "").strip()
                link  = (it.findtext("link") or "").strip()
                desc  = (it.findtext("description") or "").strip()
                pub_link = self._extract_publisher_from_description(desc)
                best = pub_link or self._extract_original_from_link(link)
                items.append({"title": title, "link": best})
                if len(items) >= max_items: break
        except Exception as e:
            print(f"Error parsing RSS for {topic_key}: {e}")

        if not items:
            print(f"Warning: no news found for {topic_key}, using fallback.")
            items = [{"title": f"Current industry trends in {topic_key.replace('_',' ')}", "link": ""}]
        return items

    def pick_publisher_link(self, news_items: List[dict]) -> str:
        # prefer allow-list
        candidates = []
        for it in news_items:
            l = (it.get("link") or "").strip()
            if not l or is_google_news(l): continue
            parsed = urlparse(l)
            if parsed.scheme.startswith("http") and parsed.netloc and parsed.path and parsed.path != "/":
                candidates.append(l)
        if not candidates: return ""
        # prefer allowed publishers first
        for l in candidates:
            if domain(l) in ALLOWED_PUBLISHERS:
                return l
        return candidates[0]

    # ---------- prompting ----------
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
            "story": "1) short scene/opening, 2) what changed, 3) concrete lesson for partnerships/gtm, 4) one-line CTA.",
            "tactical": "1) problem, 2) 3 bullet-style tactics in prose, 3) expected outcome, 4) CTA.",
            "contrarian": "1) common belief, 2) why it fails, 3) better pattern, 4) example, 5) CTA.",
            "data_point": "1) single metric or observation, 2) implication, 3) one play to run, 4) CTA.",
            "playbook": "1) context, 2) simple playbook steps, 3) risk to avoid, 4) CTA.",
        }
        tone_line = ("Tone: inspirational with restraint, never chest-beating."
                     if tone == "inspirational" else "Tone: thought leadership, specific and useful.")
        return (
            f"{persona} {tone_line} Topic: {topic_key.replace('_',' ')}. {structures.get(style,'playbook')}\n\n"
            "Requirements:\n"
            f"- Strictly {WORD_MIN}–{WORD_MAX} words. Vary sentence length and rhythm.\n"
            "- Avoid buzzwords. No generic openers like 'As a'. No em dashes. No semicolons.\n"
            "- Use a clean, confident, human voice. Keep it practical.\n"
            "- End with exactly 3 relevant hashtags on a final separate line.\n"
            f"- {link_instruction}\n\n"
            f"Recent items to anchor context:\n{news_context}\n\n"
            "Return only the post text."
        )

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
                if any(texts): return "\n".join(t for t in texts if t)
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
            if any(texts): return "\n".join(t for t in texts if t)
        if "text" in cand: return cand["text"]
        return None

    def generate_post_with_gemini(self, topic_key: str, news_items: List[dict],
                                  include_link: bool, tone: str, style: str) -> str | None:
        if not self.gemini_key: return None
        attempts = [
            {"api": "v1", "model": "gemini-2.5-flash-lite", "keep": 2, "max_out": 520},
            {"api": "v1", "model": "gemini-2.5-flash",      "keep": 2, "max_out": 1200},
        ]
        headers = {"Content-Type": "application/json"}
        for i, step in enumerate(attempts, start=1):
            base = f"https://generativelanguage.googleapis.com/{step['api']}/models/{step['model']}:generateContent"
            prompt = self._build_prompt(topic_key, news_items, include_link, tone, style, keep=step["keep"])
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": step["max_out"]}}
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
                    fr  = data.get("candidates", [{}])[0].get("finishReason")
                    ttc = data.get("usageMetadata", {}).get("thoughtsTokenCount", 0)
                    if fr == "MAX_TOKENS":
                        bigger = max(step["max_out"], ttc + 400, 1400)
                        body["generationConfig"]["maxOutputTokens"] = bigger
                        print(f"Retrying {step['model']} with maxOutputTokens={bigger} (thinking={ttc})")
                        resp = requests.post(f"{base}?key={self.gemini_key}", headers=headers, json=body, timeout=45)
                        if resp.status_code == 200:
                            text = self._extract_text_from_gemini(resp.json())
                            if text: return self.enforce_style_rules(self.debuzz(text))
                    continue
                return self.enforce_style_rules(self.debuzz(text))
            except Exception as e:
                print(f"Error generating post on attempt {i}: {e}")
                continue
        return None

    # ---------- OpenAI fallback (optional) ----------
    def generate_post_with_openai(self, topic_key: str, news_items: List[dict],
                                  include_link: bool, tone: str, style: str) -> str | None:
        if not self.openai_key: return None
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
            f"Tone: {tone_line}. Topic: {topic_key.replace('_',' ')}. Structure: {structures.get(style, 'playbook')}.\n\n"
            f"Requirements:\n- Strictly {WORD_MIN}–{WORD_MAX} words. Vary sentence length and rhythm.\n"
            "- Direct, practical, human. Avoid buzzwords. No generic openers. No em dashes. No semicolons.\n"
            "- End with exactly 3 relevant hashtags on the last line.\n"
            f"- {link_instruction}\n\nRecent items:\n{news_context}\n\nReturn only the post text."
        )
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7, "max_tokens": 520}
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

    # ---------- refinement ----------
    def expand_if_short(self, text: str, min_words: int = WORD_MIN, target_high: int = WORD_MAX) -> str:
        if not text or self.word_count(text) >= min_words:
            return text
        prompt = (
            f"Expand and refine the LinkedIn post to approximately {min_words}-{target_high} words. "
            "Keep the same ideas, facts, and any URL. Add one concrete example or detail. "
            "Avoid hype. No em dashes. No semicolons. End with the existing hashtags line if present; "
            "if not present, do not add hashtags.\n\nPost:\n" + text
        )
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent"
                body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 420}}
                r = requests.post(f"{url}?key={self.gemini_key}", headers={"Content-Type": "application/json"},
                                  json=body, timeout=35)
                if r.status_code == 200:
                    expanded = self._extract_text_from_gemini(r.json())
                    if expanded:
                        return self.enforce_style_rules(self.debuzz(expanded))
            if self.openai_key:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3, "max_tokens": 420}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                  json=body, timeout=35)
                if r.status_code == 200:
                    expanded = r.json()["choices"][0]["message"]["content"]
                    if expanded:
                        return self.enforce_style_rules(self.debuzz(expanded))
        except Exception:
            pass
        return text

    def polish_with_model(self, text: str) -> str:
        if not text: return text
        prompt = (
            "Edit the LinkedIn post to improve grammar, flow, and clarity. "
            "Keep the same ideas and any URL. Remove any hashtags you see. "
            "Neutral, senior voice. No em dashes. No semicolons. Return only the edited text.\n\n"
            f"Post:\n{text}"
        )
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent"
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
                body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2, "max_tokens": 320}
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                  json=body, timeout=35)
                if r.status_code == 200:
                    edited = r.json()["choices"][0]["message"]["content"]
                    if edited:
                        return self.enforce_style_rules(self.debuzz(edited))
        except Exception:
            pass
        return self.enforce_style_rules(self.debuzz(text))

    # ---------- hashtags & finalize ----------
    def _keywords_from_titles(self, news_items: List[dict], k: int = 6) -> List[str]:
        text = " ".join((it.get("title") or "") for it in news_items[:4])
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
        freq: Dict[str, int] = {}
        for w in words:
            wl = w.lower()
            if wl in self.STOPWORDS or len(wl) < 4: continue
            freq[wl] = freq.get(wl, 0) + 1
        keys = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
        def camel(t: str) -> str: return re.sub(r"[^A-Za-z0-9]", "", t.title())
        return [camel(t[0]) for t in keys if t[0]]

    def _normalize_tag(self, tag: str) -> str:
        low = tag.lower()
        if low in {"#agent", "#agents"}: return "#AIAgents"
        if low == "#cx": return "#CustomerExperience"
        return tag

    def curated_hashtags(self, topic_key: str, body: str,
                         news_items: List[dict], state: dict) -> str:
        base = [self._normalize_tag(t) for t in self.topic_tags.get(topic_key, ["#B2B", "#GTM"])]
        dynamic = ["#" + k for k in self._keywords_from_titles(news_items, k=8) if k]
        dynamic = [self._normalize_tag("#" + re.sub(r"[^A-Za-z0-9]", "", d)) for d in dynamic]
        pool = list(dict.fromkeys(base + dynamic))

        body_lower = body.lower()
        def is_ok(tag: str) -> bool:
            tl = tag.lower()
            if tl in self.BANNED_BRAND_TAGS and tl[1:] not in body_lower: return False
            return True

        candidates = [t for t in pool if is_ok(t) and t.lower() not in body_lower]
        recent_sets = state.get("recent_hashtag_sets", [])
        random.shuffle(candidates)
        chosen = None
        for i in range(min(20, len(candidates))):
            trio = candidates[i:i+3]
            if len(trio) < 3: break
            trial = " ".join(trio)
            if trial and trial.lower() not in recent_sets:
                chosen = trio; break
        if not chosen:
            fallback = [t for t in base if t.lower() not in self.BANNED_BRAND_TAGS]
            while len(fallback) < 3:
                for g in ["#B2B", "#GTM", "#AI", "#EnterpriseAI"]:
                    if g not in fallback: fallback.append(g)
                    if len(fallback) == 3: break
            chosen = fallback[:3]
        result = " ".join(chosen)
        recent_sets.append(result.lower())
        state["recent_hashtag_sets"] = recent_sets[-10:]
        return result

    def inject_open_close(self, body: str, state: dict) -> Tuple[str, str]:
        op = random.choice(self.openers)
        recent = state.get("recent_closers", [])
        random.shuffle(self.closers)
        closer = next((c for c in self.closers if c.lower() not in recent), random.choice(self.closers))
        recent.append(closer.lower())
        state["recent_closers"] = recent[-10:]
        new_body = f"{op} {body}".strip()
        return new_body, closer

    def dedupe_sentences(self, text: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        seen, out = set(), []
        for s in parts:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key); out.append(s.strip())
        return " ".join(out)

    def _pad_to_minimum(self, text: str, topic_key: str) -> str:
        """If still under WORD_MIN, append concrete, topic-specific detail sentences (before link/hashtags)."""
        pads = {
            "tech_partnerships": [
                "Write the joint KPI at the top of every weekly note.",
                "If the milestone changes, capture why, then reset the baseline together."
            ],
            "ai": [
                "Choose inputs you can audit and fix quickly.",
                "Track acceptance rate and rework time as first-class metrics."
            ],
            "payments": [
                "Watch authorization rate, fraud loss, and settlement timing as one dashboard.",
                "Run a small A/B at the network edge before you rewire checkout."
            ],
            "agentic_commerce": [
                "Constrain the agent with budget, catalog, and escalation rules.",
                "Cache decisions that users accept to make the next session faster."
            ],
            "generative_ai": [
                "Template the edits so reviewers can approve in a click.",
                "Keep a short list of blocked sources and phrases."
            ],
            "ai_automations": [
                "Include prior steps, inputs, and confidence when you escalate.",
                "Define stop conditions up front so the bot knows when to hand off."
            ],
        }
        while self.word_count(text) < WORD_MIN:
            extra = random.choice(pads.get(topic_key, pads["tech_partnerships"]))
            if not text.endswith("."): text += "."
            text += " " + extra
            if self.word_count(text) > WORD_MAX + 10: break
        return text

    def sanitize_and_finalize(self, body: str, topic_key: str, include_link: bool,
                              news_items: List[dict], state: dict) -> str:
        # strip any hashtag lines the model may have added
        lines = [ln for ln in body.splitlines() if not re.fullmatch(r'\s*(#\w+\s*){2,}\s*', ln.strip())]
        body = "\n".join(lines)
        body = re.sub(r'(?<!\w)#\w+', "", body)   # remove inline stray hashtags
        body = self.enforce_style_rules(self.debuzz(body)).strip()

        # remove trailing hashtags if present
        m = re.search(r'(^|\n)\s*(#\w+(?:\s+#\w+){1,})\s*$', body, flags=re.IGNORECASE | re.MULTILINE)
        if m: body = body[: m.start()].rstrip()

        # expand to target band, dedupe, and ensure closing punctuation
        body = self.expand_if_short(body, min_words=WORD_MIN, target_high=WORD_MAX)
        body = self.dedupe_sentences(body)
        if not self.ends_cleanly(body): body = body.rstrip(' "\n') + "."

        # opener & closer
        body, closer = self.inject_open_close(body, state)
        if not self.ends_cleanly(body): body += " "
        body += closer
        if not self.ends_cleanly(body): body += "."

        # last-mile pad-to-minimum *before* link/hashtags
        body = self._pad_to_minimum(body, topic_key)

        # link
        link_line = ""
        if include_link:
            good = self.pick_publisher_link(news_items)
            if good:
                link_line = f"\n\n{good}"

        # hashtags
        tags = self.curated_hashtags(topic_key, body, news_items, state)
        hashtags_line = f"\n\n{tags}" if tags else ""

        final_text = body + link_line + hashtags_line
        return self.enforce_style_rules(self.debuzz(final_text)).strip()

    def quality_gate(self, text: str, include_link: bool) -> Tuple[bool, str]:
        if not text: return False, "empty"
        wc = self.word_count(text)
        if wc < WORD_MIN: return False, f"too short ({wc} words)"
        if wc > WORD_MAX + 20: return False, f"too long ({wc} words)"
        if len(re.findall(r"[.!?]", text)) < 3: return False, "too few sentences"
        lines = [l for l in text.splitlines() if l.strip()]
        if lines:
            last = lines[-1].strip()
            if "#" in text and not re.fullmatch(r"(#\w+\s*){2,}", last):
                return False, "hashtags not at end as block"
        if include_link and "news.google.com" in text:
            return False, "google news link present"
        return True, "ok"

    # ---------- LinkedIn ----------
    def post_to_linkedin(self, text: str) -> bool:
        if not self.linkedin_token or not self.person_urn:
            print("Missing LinkedIn token or Person URN. Cannot post."); return False
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
                print("Posted to LinkedIn"); return True
            print(f"Post failed: {r.status_code} - {r.text[:400]}")
            if r.status_code == 401: print("Hint: token expired or missing w_member_social scope.")
            if r.status_code == 403: print("Hint: app lacks member post permission.")
            return False
        except Exception as e:
            print(f"LinkedIn error: {e}"); return False

    # ---------- Local compose (richer) ----------
    def local_compose(self, topic_key: str) -> str:
        themes = {
            "tech_partnerships": {
                "hook": "Great partnerships feel simple to customers because the hard work is invisible.",
                "angle": "clarity of ownership and a shared metric.",
                "example": "When we co-sold with a data platform, we published one KPI—‘activated accounts in 30 days’. Sales, CS, and the partner team met weekly against that number.",
                "tip": "Define the milestone, expose it in both dashboards, and agree on the next experiment before the meeting ends."
            },
            "ai": {
                "hook": "AI wins in the enterprise when it reduces time to value, not just cost.",
                "angle": "choose narrow problems with clean data and visible owners.",
                "example": "A small claims triage bot cut backlog by 18% in six weeks because the inputs were standardized and the handoffs were scripted.",
                "tip": "Pick one workflow, write the ‘before/after’, and automate the boring middle first."
            },
            "payments": {
                "hook": "Payments is a trust business. Speed and acceptance matter; proof beats promises.",
                "angle": "instrument risk, auth rates, and time-to-first-settlement.",
                "example": "A merchant moved retries to the network edge and lifted approvals by 140 bps without touching checkout.",
                "tip": "Start with three dials: authorization rate, fraud loss, and settlement timing—then publish them weekly."
            },
            "agentic_commerce": {
                "hook": "Agentic commerce turns browsing into doing by removing decisions, not just clicks.",
                "angle": "let the system propose the next best action with guardrails.",
                "example": "A concierge bot pre-fills size and delivery constraints and surfaces two in-stock bundles; abandonment falls because the choice set is sane.",
                "tip": "Autonomy needs constraints: budget, catalog, and escalation rules—write them down first."
            },
            "generative_ai": {
                "hook": "Generative AI helps when the inputs are reliable and the edits are cheap.",
                "angle": "tight loops with human accept/reject signals.",
                "example": "Legal review cut drafting time by 35% using clause templates and a redline diff users could reject in one click.",
                "tip": "Instrument acceptance rate and rework time; if they stall, fix the prompts or the data—not the marketing."
            },
            "ai_automations": {
                "hook": "Automation shines when it owns the boring work and hands off the hard parts clearly.",
                "angle": "route exceptions with context, not tickets.",
                "example": "Escalations included prior steps, inputs, and confidence; resolution time halved because rework disappeared.",
                "tip": "Design the ‘stop conditions’ first. Automation without an exit is a support queue."
            },
        }
        d = themes.get(topic_key, themes["tech_partnerships"])
        lines = [d["hook"], f"The lever is {d['angle']}", d["example"], d["tip"], random.choice(INSIGHT_VARIATIONS)]
        return self.enforce_style_rules(self.debuzz(" ".join(lines)))

    # ---------- Main ----------
    def run_weekly_post(self) -> None:
        print("\n" + "=" * 60)
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        state = self.load_state()
        if not self.should_post_today(state): return
        random_delay_minutes(0, 120)

        topic_key, state = self.get_next_topic(state)
        include_link = True if os.getenv("REQUIRE_LINK") == "1" else (random.random() < 0.6)
        tone = random.choice(["inspirational", "thought_leadership"])
        style = random.choice(self.style_modes)

        print(f"Topic: {topic_key} | tone: {tone} | style: {style} | link: {include_link}")
        q1 = random.choice(self.topics[topic_key])
        print(f"Fetching news for: {q1}")
        news_items = self.fetch_news(topic_key, q1)
        print(f"Found {len(news_items)} items")

        # If we want a link but couldn't find a good one, try a second query
        have_deep = any((it.get("link") and not is_google_news(it["link"])) for it in news_items)
        if include_link and not have_deep:
            q2 = random.choice([q for q in self.topics[topic_key] if q != q1]) if len(self.topics[topic_key]) > 1 else q1
            print(f"No clean publisher link found; retrying news fetch for deep link: {q2}")
            news_items2 = self.fetch_news(topic_key, q2)
            if any((it.get("link") and not is_google_news(it["link"])) for it in news_items2):
                news_items = news_items2
            else:
                print("Still no clean publisher link after retry.")
                if os.getenv("REQUIRE_LINK") == "1":
                    print("REQUIRE_LINK=1 set — will still proceed linkless if none is safe.")
                include_link = False

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
        post_body = self.polish_with_model(post_body)
        final_text = self.sanitize_and_finalize(post_body, topic_key, include_link, news_items, state)

        ok_q, reason = self.quality_gate(final_text, include_link)
        if not ok_q:
            print(f"Quality gate failed: {reason}. Falling back to local compose.")
            fallback = self.local_compose(topic_key)
            fallback = self.polish_with_model(fallback)
            final_text = self.sanitize_and_finalize(fallback, topic_key, include_link, news_items, state)

        print("\nGenerated post\n" + "-" * 60 + f"\n{final_text}\n" + "-" * 60)

        ok = self.post_to_linkedin(final_text)
        if ok:
            state["post_count"] = state.get("post_count", 0) + 1
            state["last_post_date"] = datetime.now().strftime("%Y-%m-%d")
            self.save_state(state)
            print(f"Success. Total posts: {state['post_count']}")
        else:
            print("Publish failed")

    # remove repetitive “playbook” endings if present
    def _strip_repeated_playbook(self, text: str) -> str:
        if not text: return text
        patterns = [
            r"Start with one use case.*?(weekly|week)\.",
            r"Focus on one measurable outcome.*?(repeatable|scale)\.",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s{2,}", " ", text).strip()


if __name__ == "__main__":
    LinkedInAIAgent().run_weekly_post()
