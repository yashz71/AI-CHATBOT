# agent/workflow.py
from langgraph.graph import StateGraph, END, START

from agent.utils.nodes import supervisor_router, route_supervisor
from agent.utils.state import AgentState


def build_workflow(general_subgraph, finance_agent):

    workflow = StateGraph(AgentState)

    # =========================================================
    # NODES
    # =========================================================

    workflow.add_node(
        "supervisor",
        supervisor_router
    )

    workflow.add_node(
        "finance_agent",
        finance_agent
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
            "finance_agent": "finance_agent",

            "historical_data_subgraph":
                "historical_data_subgraph"

        }
    )

    # =========================================================
    # SUBGRAPH RETURN
    # =========================================================

    workflow.add_edge("finance_agent", END)
    workflow.add_edge("historical_data_subgraph", END)

    return workflow

# =========================================================
# COMPILE
# =========================================================

