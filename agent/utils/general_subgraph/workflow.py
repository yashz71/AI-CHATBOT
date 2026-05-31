# agent/workflow.py
from langgraph.graph import StateGraph, END, START

from agent.utils.general_subgraph.nodes import tool_node, should_continue, call_model
from agent.utils.state import AgentState


def build_finance_agent():

    workflow = StateGraph(AgentState)

    # =========================================================
    # NODES
    # =========================================================

    workflow.add_node(
        "finance_agent",
        call_model
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    # =========================================================
    # START
    # =========================================================

    workflow.add_edge(
        START,
        "finance_agent"
    )

    # =========================================================
    # NORMAL AGENT LOOP
    # =========================================================

    workflow.add_conditional_edges(
        "finance_agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge(
        "tools",
        "finance_agent"
    )

    return workflow.compile()

# =========================================================
# COMPILE
# =========================================================

