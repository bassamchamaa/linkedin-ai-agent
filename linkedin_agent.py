import os
import json
import requests
from datetime import datetime, timedelta
import random

class LinkedInAIAgent:
    def __init__(self):
        self.linkedin_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        self.openai_key = os.getenv('OPENAI_API_KEY')  # or ANTHROPIC_API_KEY
        self.person_urn = os.getenv('LINKEDIN_PERSON_URN')
        
        self.topics = {
            'tech_partnerships': [
                'technology partnerships business development',
                'B2B tech strategic partnerships',
                'enterprise technology alliances'
            ],
            'ai': [
                'artificial intelligence enterprise',
                'AI business applications',
                'machine learning partnerships'
            ],
            'payments': [
                'fintech payments partnerships',
                'payment technology trends',
                'digital payments innovation'
            ]
        }
        
        # Track which topic was posted last to ensure rotation
        self.state_file = 'agent_state.json'
    
    def fetch_news(self, topic, query):
        """Fetch recent news using Google News RSS (free, no API key needed)"""
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = requests.get(rss_url, timeout=10)
            if response.status_code == 200:
                # Parse RSS feed (simple extraction)
                items = []
                content = response.text
                
                # Extract titles and links (basic parsing)
                import re
                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
                links = re.findall(r'<link>(.*?)</link>', content)
                
                for title, link in zip(titles[1:6], links[1:6]):  # Skip first (feed title)
                    items.append({'title': title, 'link': link})
                
                return items
        except Exception as e:
            print(f"Error fetching news for {topic}: {e}")
            return []
    
    def generate_post_with_anthropic(self, topic, news_items, include_link):
        """Generate LinkedIn post using Claude API"""
        news_context = "\n".join([f"- {item['title']}: {item['link']}" for item in news_items[:3]])
        
        link_instruction = ""
        if include_link:
            link_instruction = "You MUST include a link to one of the news articles in your post."
        else:
            link_instruction = "Do NOT include any links. This should be a thought leadership piece based on current trends."
        
        prompt = f"""You are a senior sales leader in tech and fintech with deep expertise in strategic partnerships. Create a LinkedIn post about {topic.replace('_', ' ')}. 

Recent news/trends:
{news_context}

Requirements:
- 150-250 words
- Write from the perspective of a senior partnerships executive
- Professional but conversational, approachable tone
- Share strategic insights about partnerships, deal-making, or go-to-market strategies
- {link_instruction}
- Include 2-3 relevant hashtags at the end
- Make it engaging and provide real value
- Share a strong point of view or actionable insight
- CRITICAL: Do not use em dashes (—) anywhere in the post. Use commas, periods, or colons instead.

Format: Just return the post text, ready to publish."""

        headers = {
            'x-api-key': self.openai_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        
        data = {
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 600,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                post = response.json()['content'][0]['text']
                # Remove any em dashes that might appear
                post = post.replace('—', ',').replace('–', ',')
                return post
        except Exception as e:
            print(f"Error generating post: {e}")
        
        return None
    
    def generate_post_with_openai(self, topic, news_items, include_link):
        """Generate LinkedIn post using OpenAI API"""
        news_context = "\n".join([f"- {item['title']}: {item['link']}" for item in news_items[:3]])
        
        link_instruction = ""
        if include_link:
            link_instruction = "You MUST include a link to one of the news articles in your post."
        else:
            link_instruction = "Do NOT include any links. This should be a thought leadership piece based on current trends."
        
        prompt = f"""You are a senior sales leader in tech and fintech with deep expertise in strategic partnerships. Create a LinkedIn post about {topic.replace('_', ' ')}. 

Recent news/trends:
{news_context}

Requirements:
- 150-250 words
- Write from the perspective of a senior partnerships executive
- Professional but conversational, approachable tone
- Share strategic insights about partnerships, deal-making, or go-to-market strategies
- {link_instruction}
- Include 2-3 relevant hashtags at the end
- Make it engaging and provide real value
- Share a strong point of view or actionable insight
- CRITICAL: Do not use em dashes (—) anywhere in the post. Use commas, periods, or colons instead.

Format: Just return the post text, ready to publish."""

        headers = {
            'Authorization': f'Bearer {self.openai_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4o-mini',  # Free tier available
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 600,
            'temperature': 0.7
        }
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                post = response.json()['choices'][0]['message']['content']
                # Remove any em dashes that might appear
                post = post.replace('—', ',').replace('–', ',')
                return post
        except Exception as e:
            print(f"Error generating post: {e}")
        
        return None
    
    def post_to_linkedin(self, text):
        """Post to LinkedIn via API"""
        url = 'https://api.linkedin.com/v2/ugcPosts'
        
        headers = {
            'Authorization': f'Bearer {self.linkedin_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        post_data = {
            'author': f'urn:li:person:{self.person_urn}',
            'lifecycleState': 'PUBLISHED',
            'specificContent': {
                'com.linkedin.ugc.ShareContent': {
                    'shareCommentary': {
                        'text': text
                    },
                    'shareMediaCategory': 'NONE'
                }
            },
            'visibility': {
                'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC'
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=post_data, timeout=30)
            
            if response.status_code in [200, 201]:
                print(f"✓ Successfully posted to LinkedIn")
                return True
            else:
                print(f"✗ Failed to post: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"✗ Error posting to LinkedIn: {e}")
            return False
    
    def load_state(self):
        """Load the agent state to track topic rotation"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {'last_topics': [], 'post_count': 0, 'last_post_date': None}
    
    def save_state(self, state):
        """Save the agent state"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def should_post_today(self, state):
        """Check if we should post today (only once per day)"""
        last_post = state.get('last_post_date')
        today = datetime.now().strftime('%Y-%m-%d')
        
        if last_post == today:
            print(f"Already posted today ({today}). Skipping.")
            return False
        return True
    
    def get_next_topic(self, state):
        """Get the next topic ensuring rotation"""
        last_topics = state.get('last_topics', [])
        available_topics = [t for t in self.topics.keys() if t not in last_topics]
        
        # If all topics used, reset rotation
        if not available_topics:
            last_topics = []
            available_topics = list(self.topics.keys())
        
        # Pick random from available
        next_topic = random.choice(available_topics)
        
        # Update state
        last_topics.append(next_topic)
        if len(last_topics) > 2:  # Keep only last 2 topics
            last_topics = last_topics[-2:]
        
        state['last_topics'] = last_topics
        return next_topic, state
    
    def run_weekly_post(self):
        """Main function to generate and post once (called 3x per week)"""
        print(f"\n{'='*60}")
        print(f"LinkedIn AI Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Load state and check if already posted today
        state = self.load_state()
        
        if not self.should_post_today(state):
            return
        
        # Determine next topic
        topic_name, state = self.get_next_topic(state)
        state['post_count'] = state.get('post_count', 0) + 1
        state['last_post_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Randomly decide if this post should include a link (60% yes, 40% thought piece)
        include_link = random.random() < 0.6
        post_type = "with reference link" if include_link else "thought leadership piece"
        
        print(f"--- Topic: {topic_name.replace('_', ' ').title()} ({post_type}) ---\n")
        
        # Fetch news
        query = random.choice(self.topics[topic_name])
        print(f"Fetching news for: {query}")
        news_items = self.fetch_news(topic_name, query)
        
        if not news_items:
            print(f"⚠ No news found for {topic_name}, using fallback...")
            news_items = [{'title': 'Industry trends in ' + topic_name, 'link': ''}]
        
        print(f"Found {len(news_items)} news items")
        
        # Generate post
        post_text = None
        if 'ANTHROPIC_API_KEY' in os.environ:
            print("Generating post with Claude...")
            post_text = self.generate_post_with_anthropic(topic_name, news_items, include_link)
        elif 'OPENAI_API_KEY' in os.environ:
            print("Generating post with OpenAI...")
            post_text = self.generate_post_with_openai(topic_name, news_items, include_link)
        
        if not post_text:
            print(f"⚠ Failed to generate post")
            return
        
        print(f"\nGenerated post:\n{'-'*60}\n{post_text}\n{'-'*60}")
        
        # Post to LinkedIn
        success = self.post_to_linkedin(post_text)
        
        if success:
            self.save_state(state)
            print(f"\n✅ Post #{state['post_count']} published successfully!")
        else:
            print(f"\n❌ Failed to publish post")
        
        print(f"\nNext post will be: {[t for t in self.topics.keys() if t not in state['last_topics']]}")
        print(f"Posts remaining in cycle: {3 - len(state['last_topics'])}")

if __name__ == '__main__':
    agent = LinkedInAIAgent()
    agent.run_weekly_post()
