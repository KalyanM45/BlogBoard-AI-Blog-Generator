from graph.state import BlogState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from graph.nodes import (
    autonomous_topic_selection,
    llm_generate,
    save_markdown,
    update_articles_json,
)


def _should_skip(state: BlogState) -> str:
    if state.get("skipped"):
        return "skip"
    return "continue"


def build_graph() -> StateGraph:

    builder = StateGraph(BlogState)

    builder.add_node("autonomous_topic_selection", autonomous_topic_selection)
    builder.add_node("llm_generate", llm_generate)
    builder.add_node("save_markdown", save_markdown)
    builder.add_node("update_articles_json", update_articles_json)

    builder.add_edge(START, "autonomous_topic_selection")
    builder.add_conditional_edges("autonomous_topic_selection", _should_skip, {"skip": END, "continue": "llm_generate"})
    builder.add_edge("llm_generate", "save_markdown")
    builder.add_edge("save_markdown", "update_articles_json")
    builder.add_edge("update_articles_json", END)
    return builder.compile(checkpointer=InMemorySaver())

graph = build_graph()
