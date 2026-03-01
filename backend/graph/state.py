"""
state.py
Defines the BlogState TypedDict that flows through every node of the LangGraph pipeline.
"""

from typing import TypedDict


class BlogState(TypedDict, total=False):
    # ── Inputs (set by the runner before graph.invoke) ────────────────────────
    date: str          # ISO date string, e.g. "2026-03-01"
    dry_run: bool      # When True, skip LLM calls and file writes

    # ── Node 1 – consolidate_schedule ─────────────────────────────────────────
    schedule: dict     # Full schedule dict keyed by date string

    # ── Node 2 – get_domain_topic ─────────────────────────────────────────────
    domain: str        # Short domain key, e.g. "ml", "dl", "nlp"
    topic: str         # Scheduled topic string for the given date
    skipped: bool      # True when no entry exists for the given date

    # ── Node 3 – llm_generate ────────────────────────────────────────────────
    title: str
    description: str
    tags: list         # List of lowercase hyphenated tag strings
    slug: str          # URL-safe filename slug derived from title
    content: str       # Full markdown article body
    read_time: str     # Estimated read time string, e.g. "7 min"

    # ── Node 4 – save_markdown ────────────────────────────────────────────────
    md_path: str       # Absolute path where the .md file was written

    # ── Node 5 – update_articles_json ────────────────────────────────────────
    # (no new fields; result is a side effect on articles.json)
