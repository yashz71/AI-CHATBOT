# agent/workflow.py
from langgraph.graph import StateGraph, END, START

from agent.utils.nodes import supervisor_router, tool_node, route_supervisor, should_continue, call_model
from agent.utils.state import AgentState


def build_workflow(general_subgraph):

    workflow = StateGraph(AgentState)

    # =========================================================
    # NODES
    # =========================================================

    workflow.add_node(
        "supervisor",
        supervisor_router
    )

    workflow.add_node(
        "agent",
        call_model
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    workflow.add_node(
        "historical_data_subgraph",
        general_subgraph
    )

    # =========================================================
    # START
    # =========================================================

    workflow.add_edge(
        START,
        "supervisor"
    )

    # =========================================================
    # SUPERVISOR ROUTING
    # =========================================================

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "tools": "agent",

            "historical_data_subgraph":
                "historical_data_subgraph",

            "END": END,
        }
    )

    # =========================================================
    # NORMAL AGENT LOOP
    # =========================================================

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge(
        "tools",
        "agent"
    )

    # =========================================================
    # SUBGRAPH RETURN
    # =========================================================

    workflow.add_edge(
        "historical_data_subgraph",
        "agent"
    )
    return workflow

# =========================================================
# COMPILE
# =========================================================

