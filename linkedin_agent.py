"""Thin wrapper that runs the heavy LinkedIn agent implementation."""

from __future__ import annotations

from linkedin_agent_core import LinkedInAIAgent

__all__ = ["LinkedInAIAgent"]


def main() -> None:
    """Entry point used by the GitHub workflow."""
    LinkedInAIAgent().run_weekly_post()


if __name__ == "__main__":
    main()
