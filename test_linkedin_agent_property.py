# test_linkedin_agent_property.py
import re
from unittest.mock import MagicMock
from hypothesis import given, strategies as st

from linkedin_agent_refined import LinkedInAIAgent, AgentConfig


def make_agent():
    cfg = AgentConfig(dry_run=True, disable_delay=True)
    session = MagicMock()
    session.get.return_value.status_code = 200
    session.get.return_value.text = "<rss><channel></channel></rss>"
    agent = LinkedInAIAgent(cfg, session=session, rng=MagicMock(), sleeper=lambda s: None)
    agent.rng.choice.side_effect = lambda seq: seq[0]
    agent.rng.shuffle.side_effect = lambda seq: None
    return agent


@given(st.text())
def test_enforce_style_never_crash(s):
    a = make_agent()
    out = a.enforce_style_rules(s)
    assert isinstance(out, str)


@given(st.text())
def test_debuzz_idempotentish(s):
    a = make_agent()
    once = a.debuzz(s)
    twice = a.debuzz(once)
    assert twice == a.debuzz(twice)  # applying repeatedly stabilizes


@given(st.lists(st.dictionaries(keys=st.text(min_size=1, max_size=5),
                                values=st.text(), min_size=0, max_size=5),
               min_size=0, max_size=5))
def test_keywords_from_titles_no_crash(items):
    a = make_agent()
    # shape the items to include "title" sometimes
    shaped = [{"title": it.get("title", "")} for it in items]
    ks = a._keywords_from_titles(shaped, k=6)
    assert isinstance(ks, list)
    assert all(isinstance(k, str) for k in ks)


@given(st.sampled_from(["ai", "tech_partnerships"]))
def test_compose_always_has_hashtags(topic):
    a = make_agent()
    txt, meta = a.compose_post(topic_key=topic)
    last_line = txt.strip().splitlines()[-1]
    assert last_line.startswith("#")
    assert len(re.findall(r"#\w+", last_line)) >= 1
