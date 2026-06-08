from agent.utils.state import AgentState
from agent.utils.models.llm import model
from typing import Literal
from pydantic import BaseModel


class RouterDecision(BaseModel):
    """
    Determines which agent/subgraph
    should handle the request.
    """

    next_agent: Literal[
        "finance_agent",
        "historical_data_subgraph"
    ]


router_model = model.with_structured_output(
    RouterDecision
)


def supervisor_router(state: AgentState):

    messages = state["messages"]

    decision = router_model.invoke(
        f"""
        You are an orchestration supervisor
        for a financial multi-agent system.

        Decide which specialized workflow
        should handle the user request.

        Available routes:
        1. finance_agent
           - general tools
           
           - web search
           - RAG
           - generic finance tasks
           - any question not related to historical_data_subgraph
        2. historical_data_subgraph
           - requests for:
             historical prices
             OHLCV data
             stock history
             downloadable excel/csv
             time series retrieval
             sentiment analysis
        
        Conversation:
        {messages}
        """
    )

    return {
        "next_agent": decision.next_agent
    }


def route_supervisor(state: AgentState):

    return state["next_agent"]

