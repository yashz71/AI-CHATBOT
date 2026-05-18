from langgraph.graph import StateGraph, START, END

from agent.utils.state import AgentState
from agent.utils.subgraphs.nodes import (
    call_model,
    should_continue,
    verify_hist_data,
    retry_router,
)


# =========================================================
# GRAPH INITIALIZATION
# =========================================================
def build_general_subgraph(tool_node):

    general_subgraph = StateGraph(AgentState)

    # =========================================================
    # NODES
    # =========================================================

    general_subgraph.add_node("agent", call_model)
    print(f"tool_node in subgraph: {tool_node}")
    general_subgraph.add_node("tools", tool_node)

    general_subgraph.add_node(
        "verify_hist_data",
        verify_hist_data
    )

    # =========================================================
    # START EDGE
    # =========================================================

    general_subgraph.add_edge(
        START,
        "agent"
    )

    # =========================================================
    # AGENT ROUTING
    # =========================================================

    general_subgraph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )

    # =========================================================
    # AFTER TOOL EXECUTION
    # =========================================================

    general_subgraph.add_edge(
        "tools",
        "verify_hist_data"
    )

    # =========================================================
    # VERIFY / RETRY ROUTING
    # =========================================================

    general_subgraph.add_conditional_edges(
        "verify_hist_data",
        retry_router,
        {
            "call_model": "agent",
            END: END,
        }
    )
    return general_subgraph.compile()
# =========================================================
# COMPILE
# =========================================================


