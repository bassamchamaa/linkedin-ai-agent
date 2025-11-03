# LinkedIn AI Agent 🤖

Automated LinkedIn posting aligned to a **senior tech/fintech sales leader** persona. Posts draw from curated enterprise topics and news, maintain a clean 100–150 word format, and publish on a reliable cadence via GitHub Actions.

## Cadence & Workflow

- **Schedule:** Monday 7 AM ET, Wednesday 12 PM ET, Friday 7 AM ET (GitHub Actions cron).
- **Workflow file:** `.github/workflows/linkedin-agent.yml` installs Python, runs `linkedin_agent.py`, and commits the state file after each run. The wrapper is now only a few lines and imports the heavy implementation from `linkedin_agent_core.py`, so merge mistakes can’t leave a stray `python` line in the executable.
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

### Resolving “This branch has conflicts”

When a pull request shows the yellow conflict banner, GitHub is telling you that `main` has changed since you opened your branch. Fix it locally with these steps:

1. `git fetch origin` – make sure you have the latest `main` history.
2. `git checkout <your-branch>` – switch back to the branch from the pull request.
3. `git merge origin/main` (or `git rebase origin/main`) – bring the new commits into your branch.
4. Open any files marked with conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), pick the correct code, and save.
5. `git add <each-file>` once the conflicts are resolved, then `git commit` and `git push`.

After the push the banner disappears and the PR can be merged. You can also use GitHub’s “Resolve conflicts” editor if you prefer to handle steps 3–5 in the browser.

#### Hands-on walkthrough: clearing the “Fix GPT-5 Nano fallback parameters #3” conflict

The GPT-5 Nano fallback now lives in one helper method (`_call_openai_completion`), so there is only a single conflict block to keep. When GitHub shows the diff:

1. Click **Resolve conflicts** on the pull request and open `linkedin_agent_core.py` (the implementation file).
2. Delete the `<<<<<<<`, `=======`, and `>>>>>>>` markers.
3. Keep the version that defines `_call_openai_completion` and calls it from the three OpenAI fallback spots. (It contains the `max_completion_tokens` parameter and no inline `python` marker.)
4. Scroll to the bottom of the file and press **Mark as resolved**.
5. Repeat the same idea for any README conflicts: keep the longer troubleshooting section that explains how to fetch, merge, and push.
6. Click **Commit merge** to finish, then push your branch (or let GitHub update the pull request automatically).

Once the conflicts are gone, the yellow banner disappears and the branch is ready for the next review.

Happy posting! 🎯

#### “IndentationError: unindent does not match any outer indentation level” on the `python` line

The helper eliminates the old copy/paste snippet, but if you still see this error it means a stray line containing only `python` (or ``` ) survived a merge. You can fix it without touching the command line:

1. In GitHub, open the workflow failure page and click the **View workflow file** link next to `linkedin_agent_core.py`.
2. Tap the pencil icon, delete the line that contains only `python` or ``` and make sure the surrounding block starts with `body = {` indented beneath the `try`/`if` statement.
3. Scroll to the bottom, add a short commit message such as “Remove stray python line,” and press **Commit changes**.
4. Re-run the workflow.

You can also run `git restore linkedin_agent_core.py` locally if you prefer the command line. Either path removes the stray line so Python compiles cleanly.
