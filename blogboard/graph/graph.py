from langgraph.graph import StateGraph, START, END
# We no longer need InMemorySaver purely for running scripts, but we can keep it if checkpointing is desired. 
from langgraph.checkpoint.memory import InMemorySaver

from blogboard.graph.state import BlogState

from blogboard.agents.tutorial_agent.agent import tutorial_node
from blogboard.agents.news_agent.agent import news_node
from blogboard.agents.validator_agent.agent import validator_node


def _route_start(state: BlogState) -> str:
    """Decides which Generation Track to take based on requested domain."""
    if state.get("skipped"):
        return END

    if state.get("domain") == "ainews":
        return "news_agent"
    return "tutorial_agent"

def _route_after_validator(state: BlogState) -> str:
    """Supervisor logic handling the Revision loop."""
    if state.get("revision_needed"):
        # Route back to the specific generator if rejected
        if state.get("domain") == "ainews":
            return "news_agent"
        return "tutorial_agent"
    
    # If approved by Validator, the graph perfectly concludes.
    return END

def build_graph() -> StateGraph:
    builder = StateGraph(BlogState)

    # 1. Add all agents
    builder.add_node("tutorial_agent", tutorial_node)
    builder.add_node("news_agent", news_node)
    builder.add_node("validator", validator_node)

    # 2. Wire the execution edges
    builder.add_conditional_edges(
        START, 
        _route_start, 
        {
            END: END,
            "news_agent": "news_agent",
            "tutorial_agent": "tutorial_agent"
        }
    )
    
    # Both generator agents funnel down universally to the validator agent
    builder.add_edge("tutorial_agent", "validator")
    builder.add_edge("news_agent", "validator")
    
    # Validator enforces standards, looping back if corrections are demanded
    builder.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "tutorial_agent": "tutorial_agent",
            "news_agent": "news_agent",
            END: END
        }
    )

    return builder.compile(checkpointer=InMemorySaver())

# Expose compiled graph instance
graph = build_graph()
