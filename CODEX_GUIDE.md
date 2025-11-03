# LinkedIn AI Agent Overview

## Purpose
The `linkedin_agent.py` script automates creating and optionally publishing LinkedIn thought-leadership posts for a predefined persona. It rotates through business topics, pulls fresh headlines, drafts content with AI models, cleans the copy, and posts it through LinkedIn's API.

## High-level Flow
1. **Load configuration and state** – Reads API keys and flags from environment variables, plus posting history from `agent_state.json`. Missing secrets are reported so you can run in dry mode.  
2. **Decide whether to post** – Skips if a post already went out today unless `FORCE_POST=1`.  
3. **Pick a topic and research** – Chooses a topic not used recently and fetches related news via the PYMNTS scraper (for payments) or Google News RSS. Clean publisher URLs are preferred.  
4. **Draft with AI** – Prompts Gemini (primary) or OpenAI Chat Completions (fallback) to produce a 100–150 word post with a required structure, tone, and hashtag policy.  
5. **Polish and adjust** – Removes model wrappers, reworks style, pads short drafts, injects curated openers/closers, rebuilds hashtags, and enforces the word limit.  
6. **Quality gate** – Blocks drafts that fall outside word counts, lack sentences, or mishandle hashtags/links; falls back to a local template if needed.  
7. **Publish** – Calls LinkedIn's `ugcPosts` endpoint when credentials are present; otherwise logs the final copy.

## Configuration Flags
- `LINKEDIN_ACCESS_TOKEN`, `LINKEDIN_PERSON_URN`: Required to publish. Without them the script still drafts content.  
- `GEMINI_API_KEY`, `OPENAI_API_KEY`: Model keys. Gemini is tried first.  
- `NEWS_SOURCE`: Set to `pymnts` to force the PYMNTS scraper.  
- `FORCE_POST`: Set `1` to bypass the once-per-day guard.  
- `REQUIRE_LINK`: Set `1` to force inclusion of a deep publisher link (will retry news fetching once).  
- `SKIP_DELAY` or `DISABLE_DELAY`: Disable the randomized posting delay.

## Run Cadence & Workflow Hooks
- **GitHub Actions** – `.github/workflows/linkedin-agent.yml` schedules runs three times a week: Monday 11:00 UTC, Wednesday 16:00 UTC, and Friday 11:00 UTC. Each cron entry triggers the same job that installs Python 3.10, pulls secrets, and executes `linkedin_agent.py` with `FORCE_POST=1` and `SKIP_DELAY=1` so the script always attempts a post on those days.
- **Manual dispatch** – The workflow also exposes `workflow_dispatch` if you need an ad-hoc run; trigger it from the Actions tab to run the same job on demand.
- **State commits** – After the agent runs, the workflow commits any change to `agent_state.json`, allowing the cadence guard to persist between CI invocations.

## Extending or Testing
- Add or edit topic queries and hashtag pools in the `LinkedInAIAgent` initializer.
- Adjust stylistic helpers (e.g., `enforce_style_rules`, `debuzz`, `curated_hashtags`) to tweak tone or formatting.
- Modify `local_compose` templates if you need deterministic fallback copy.
- Use the included pytest suites for integration/property checks around text sanitization and output constraints.

## Offline Fallback
When both Gemini and OpenAI refuse or error, the agent assembles a deterministic draft locally via `local_compose`. The routine stitches together stored hooks, angles, examples, and tips for each topic so you still get a polished 110–140 word post without any model calls. This path also handles quality-gate failures by regenerating copy offline, ensuring you never rely on external APIs to finish a run.

