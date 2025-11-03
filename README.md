# LinkedIn AI Agent 🤖

Automated LinkedIn posting aligned to a **senior tech/fintech sales leader** persona. Posts draw from curated enterprise topics and news, maintain a clean 100–150 word format, and publish on a reliable cadence via GitHub Actions.

## Cadence & Workflow

- **Schedule:** Monday 7 AM ET, Wednesday 12 PM ET, Friday 7 AM ET (GitHub Actions cron).
- **Workflow file:** `.github/workflows/linkedin-agent.yml` installs Python, runs `linkedin_agent.py`, and commits the state file after each run.
- **Delay handling:** The script introduces a randomized delay locally, but the workflow sets `SKIP_DELAY=1` so CI executes immediately.

## What the Agent Produces

- Rotates through six topic buckets: tech partnerships, AI, payments, agentic commerce, generative AI, and AI automations.
- Pulls recent coverage from PYMNTS.com (preferred for payments) or Google News RSS with publisher link extraction.
- Applies one of five prompt structures (story, tactical, contrarian, data point, playbook) and alternates between “inspirational with restraint” and “thought leadership—specific and useful” tones.
- Ensures posts stay between **100 and 150 words**, avoid “As a…” openers, em dashes, semicolons, and finish with exactly three curated hashtags on the final line. A deep publisher link is inserted ahead of the hashtags when available.

## Required Secrets

Create these secrets in your GitHub repository settings:

| Secret | Purpose |
| --- | --- |
| `LINKEDIN_ACCESS_TOKEN` | LinkedIn REST API token with `w_member_social` scope. |
| `LINKEDIN_PERSON_URN` | LinkedIn person identifier (without the `urn:li:person:` prefix). |
| `GEMINI_API_KEY` | Google AI Studio key for the Gemini 2.5 Flash model. |
| `OPENAI_API_KEY` | (Optional) OpenAI key used with the GPT-5 Nano fallback model. |

> ℹ️ If neither `GEMINI_API_KEY` nor `OPENAI_API_KEY` is supplied the agent logs a warning and skips publishing.

## Local Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LINKEDIN_ACCESS_TOKEN=...  # etc.
python linkedin_agent.py
```

Environment flags:

- `SKIP_DELAY=1` – bypass the randomized wait window.
- `FORCE_POST=1` – ignore the once-per-day guard (useful for testing).
- `NEWS_SOURCE=pymnts` – prefer PYMNTS coverage across all topics.
- `REQUIRE_LINK=1` – force the agent to find a publisher link or fail.

## State & Monitoring

- `agent_state.json` tracks topic rotation, hashtag usage, and post counts. It is committed back to the repository by the workflow for continuity.
- Console output (in Actions logs) shows which topic, tone, structure, and article set powered the post.

## Troubleshooting

- Missing LinkedIn secrets or model keys trigger explicit console messages and the run exits without posting.
- LinkedIn API failures raise an error, ensuring the GitHub Action surfaces as failed for quick review.
- If GitHub shows “This branch has conflicts that must be resolved,” pull the latest changes from `main` and merge or rebase them into your feature branch locally, resolve any conflict markers in files such as `linkedin_agent.py`, and push the updated branch before reopening the pull request.

Happy posting! 🎯
