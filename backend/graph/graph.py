"""
graph.py
Wires the 5-node LangGraph StateGraph for BlogBoard's article generation pipeline.

Graph shape:
    START
      ↓
    consolidate_schedule      (Node 1) — merge input JSONs → schedule.json
      ↓
    get_domain_topic          (Node 2) — pick today's entry from the schedule
      ↓  ──(skipped=True)──→  END
    llm_generate              (Node 3) — one Groq call → all article fields
      ↓
    save_markdown             (Node 4) — write .md file to frontend/blogs/
      ↓
    update_articles_json      (Node 5) — update articles.json
      ↓
    END
"""

from langgraph.graph import StateGraph, START, END

from graph.state import BlogState
from graph.nodes import (
    consolidate_schedule,
    get_domain_topic,
    llm_generate,
    save_markdown,
    update_articles_json,
)


# ─────────────────────────────────────────────────────────────────────────────

def _should_skip(state: BlogState) -> str:
    """Conditional router: exit early when no article is scheduled for this date."""
    if state.get("skipped"):
        return "skip"
    return "continue"


def build_graph() -> StateGraph:
    """Build and compile the BlogBoard LangGraph pipeline."""

    builder = StateGraph(BlogState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("consolidate_schedule",  consolidate_schedule)
    builder.add_node("get_domain_topic",      get_domain_topic)
    builder.add_node("llm_generate",          llm_generate)
    builder.add_node("save_markdown",         save_markdown)
    builder.add_node("update_articles_json",  update_articles_json)

    # ── Linear edges ───────────────────────────────────────────────────────
    builder.add_edge(START,                    "consolidate_schedule")
    builder.add_edge("consolidate_schedule",   "get_domain_topic")

    # After Node 2: conditional – skip if no schedule entry
    builder.add_conditional_edges(
        "get_domain_topic",
        _should_skip,
        {
            "skip":     END,
            "continue": "llm_generate",
        },
    )

    builder.add_edge("llm_generate",           "save_markdown")
    builder.add_edge("save_markdown",          "update_articles_json")
    builder.add_edge("update_articles_json",   END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
#  Expose compiled graph at module level (used by run.py)
# ─────────────────────────────────────────────────────────────────────────────

graph = build_graph()
