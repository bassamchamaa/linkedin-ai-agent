"""Entry wrapper for the LinkedIn AI agent.

This file intentionally stays tiny so that merge-conflict markers or stray
clipboard artifacts (like a lone ``python`` line from Markdown code blocks)
can't wedge themselves into the workflow entry point. The real implementation
lives in :mod:`linkedin_agent_core`.
"""

from __future__ import annotations

from linkedin_agent_core import LinkedInAIAgent

__all__ = ["LinkedInAIAgent", "main"]


def main() -> None:
    """Run the weekly posting routine used by the GitHub workflow."""
    LinkedInAIAgent().run_weekly_post()


if __name__ == "__main__":
    main()
