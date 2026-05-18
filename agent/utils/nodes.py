from langgraph.graph import END
from agent.utils.state import AgentState
from agent.utils.models.llm import model
from agent.utils.tools.search import search_tool
from agent.utils.tools.rag import finance_knowledge_base
from agent.utils.prompts import wealth_agent_prompt
from langgraph.prebuilt import ToolNode
from typing import Literal
from pydantic import BaseModel


class RouterDecision(BaseModel):
    """
    Determines which agent/subgraph
    should handle the request.
    """

    next_agent: Literal[
        "tools",
        "historical_data_subgraph",
        "END"
    ]


router_model = model.with_structured_output(
    RouterDecision
)

# 1. Define the Tools and the ToolNode
tools = [search_tool, finance_knowledge_base]
tool_node = ToolNode(tools)

# 2. Bind the tools to the model
# This tells the LLM "You have these specific functions available"
model_with_tools = model.bind_tools(tools)


def supervisor_router(state: AgentState):

    messages = state["messages"]

    decision = router_model.invoke(
        f"""
        You are an orchestration supervisor
        for a financial multi-agent system.

        Decide which specialized workflow
        should handle the user request.

        Available routes:

        1. tools
           - general tools
           - web search
           - RAG
           - generic finance tasks

        2. historical_data_subgraph
           - requests for:
             historical prices
             OHLCV data
             stock history
             downloadable excel/csv
             time series retrieval

        3. END
           - if no action needed

        Conversation:
        {messages}
        """
    )

    return {
        "next_agent": decision.next_agent
    }


def route_supervisor(state: AgentState):

    return state["next_agent"]


# 3. Define the Node Functions
def call_model(state: AgentState):
    """Passes the current state (history + summary) to the LLM."""
    agent_chain = wealth_agent_prompt | model_with_tools
    # We pass the summary into the prompt template
    return {"messages": [agent_chain.invoke({
        "messages": state["messages"],
        "summary": state.get("summary", "No previous summary available.")
    })]}


def should_continue(state: AgentState):
    """Determines if the LLM wants to call a tool or finish."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tools"
