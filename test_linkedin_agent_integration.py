# test_linkedin_agent_integration.py
import json
from unittest.mock import MagicMock
from linkedin_agent_refined import LinkedInAIAgent, AgentConfig


def mock_rss_with_diverse_types():
    # Titles include ints, floats, punctuation to simulate variety
    titles = [
        "AI wins 2x faster in 2025: 3.14 reasons that matter",
        "Partnership revenue +47% QoQ; playbooks that scale",
        "Why LLM latency (95th=1.2s) changes adoption"
    ]
    items = "".join(
        f'<item><title>{t}</title><link>https://news.google.com/articles/x?url=https%3A%2F%2Fpub.com%2Fart{i}</link>'
        f'<description><![CDATA[<a href="https://pub.com/art{i}">deep</a>]]></description></item>'
        for i, t in enumerate(titles)
    )
    return f"<rss><channel>{items}</channel></rss>"


def test_end_to_end_compose_and_publish_success(tmp_path, monkeypatch):
    # Arrange session: RSS OK, POST OK
    session = MagicMock()
    session.get.return_value.status_code = 200
    session.get.return_value.text = mock_rss_with_diverse_types()

    post_resp = MagicMock()
    post_resp.status_code = 201
    session.post.return_value = post_resp

    cfg = AgentConfig(
        dry_run=False, disable_delay=True,
        linkedin_token="t", person_urn="u"
    )
    agent = LinkedInAIAgent(cfg, session=session)
    agent.state_file = tmp_path / "state.json"

    # Act
    text, meta = agent.compose_post(topic_key="ai", include_link=True)
    ok, msg = agent.publish_to_linkedin(text)

    # Assert
    assert ok and msg in ("ok",)  # state saved
    saved = json.loads(open(agent.state_file).read())
    assert saved["post_count"] >= 1
    assert "pub.com" in text  # deep link
