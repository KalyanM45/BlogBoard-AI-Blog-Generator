from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import BlogState
from graph.nodes_ainews import fetch_news, llm_generate_ainews
from graph.nodes import save_markdown, update_articles_json

def build_ainews_graph() -> StateGraph:
    builder = StateGraph(BlogState)

    builder.add_node("fetch_news", fetch_news)
    builder.add_node("llm_generate_ainews", llm_generate_ainews)
    builder.add_node("save_markdown", save_markdown)
    builder.add_node("update_articles_json", update_articles_json)

    builder.add_edge(START, "fetch_news")
    builder.add_edge("fetch_news", "llm_generate_ainews")
    builder.add_edge("llm_generate_ainews", "save_markdown")
    builder.add_edge("save_markdown", "update_articles_json")
    builder.add_edge("update_articles_json", END)
    
    return builder.compile(checkpointer=InMemorySaver())

ainews_graph = build_ainews_graph()
