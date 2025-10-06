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
        self.linkedin_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()
        self.person_urn = os.getenv("LINKEDIN_PERSON_URN", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        self.topics = {
            "tech_partnerships": [
                "technology partnerships strategic alliances",
                "B2B tech ecosystem partnerships",
                "enterprise software partnerships",
            ],
            "ai": [
                "AI enterprise adoption trends",
                "artificial intelligence business transformation",
                "machine learning partnerships technology",
            ],
            "payments": [
                "fintech payments innovation trends",
                "digital payments technology partnerships",
                "payment processing innovation",
            ],
        }

        self.state_file = "agent_state.json"

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"last_topics": [], "post_count": 0, "last_post_date": None}

    def save_state(self, state):
        with open(self.state_file, "w") as f:
            json.dump(state, f)

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

    def resolve_google_news_url(self, link):
        """Extract actual publisher URL from Google News redirect"""
        try:
            if not
