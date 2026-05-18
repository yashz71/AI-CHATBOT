import pandas as pd
from datetime import datetime
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import os
from langgraph.graph import END
from langchain_core.messages import AIMessage

from agent.utils.subgraphs.prompts import agent_prompt
import agent.startup.mcp as mcp


class SubgraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

    retries: int
    retry_required: bool

    ticker: str

    hist_data: Optional[str]

    excel_file: Optional[str]
    error: Optional[str]
    download_url: str


# =========================================================
# LLM NODE
# =========================================================
def call_model(state: SubgraphState):
    """
    Calls the LLM with conversation state.

    The model decides whether:
    - it should call the historical data MCP tool
    - retry data retrieval
    - or return a failure response
    """

    print("--- HISTORICAL DATA AGENT ACTIVE ---")

    agent_chain = agent_prompt | mcp.hist_model_with_tools

    response = agent_chain.invoke({
        "messages": state["messages"],
        "summary": state.get(
            "summary",
            "No previous summary available."
        )
    })

    return {
        "messages": [response]

    }


# =========================================================
# ROUTER AFTER MODEL
# =========================================================

def should_continue(state: SubgraphState):
    """
    Determines next step after the model response.

    Routes to:
    - tools -> if the model requested a tool call
    - END -> if the model already answered directly
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


# =========================================================
# VERIFY TOOL OUTPUT NODE
# =========================================================
def extract_text(message):
    content = message.content

    if isinstance(content, list):
        content = content[0].get("text", "")

    return content


def verify_hist_data(state: SubgraphState):
    """
    Verifies whether the MCP historical data tool
    returned a valid DataFrame.

    Cases:
    1. Empty DataFrame:
       - increment retry counter
       - ask model to retry or fail gracefully

    2. Valid DataFrame:
       - convert to Excel
       - return path to parent graph
    """

    print("--- VERIFYING HISTORICAL DATA ---")

    retries = state.get("retries", 0)
    messages = state["messages"]

    # Get last tool message
    last_message = messages[-1]
    # Expected:
    # tool result stored in state["hist_data"]
    raw = extract_text(last_message)
    hist = pd.read_json(raw)
    print("last m", last_message)
    print("histo: ", hist)
    # =====================================================
    # CASE 1 -> EMPTY DATAFRAME
    # =====================================================

    if hist is None or hist.empty:
        print("--- EMPTY DATAFRAME DETECTED ---")

        if retries >= 2:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Unable to retrieve historical "
                            "market data at the moment."
                        )
                    )
                ],
                "retries": retries,
                "excel_file": None
            }

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Historical data retrieval failed. "
                        "Retrying data fetch."
                    )
                )
            ],
            "retries": retries + 1,
            "retry_required": True
        }

    # =====================================================
    # CASE 2 -> VALID DATAFRAME
    # =====================================================

    print("--- VALID DATAFRAME RECEIVED ---")

    ticker = state.get("ticker", "asset")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    EXPORT_DIR = "exports"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    filename = (
        f"{ticker}_historical_data_{timestamp}.xlsx"
    )

    # Full file path
    filepath = os.path.join(EXPORT_DIR, filename)

    # Export Excel file
    hist.reset_index().to_excel(
        filepath,
        index=False
    )
    BASE_URL = "http://localhost:8000"

    download_url = (
        f"{BASE_URL}/download/{filename}"
    )
    print(f"--- EXCEL FILE CREATED: {filename} ---")

    return {
        "excel_file": filename,
        "hist_data": hist.to_json(),
        "download_url": download_url,
        "retry_required": False,
        "messages": [
            AIMessage(
                content=(
                    f"Historical data successfully "
                    f"Download: {download_url}"
                )
            )
        ]
    }


# =========================================================
# RETRY ROUTER
# =========================================================

def retry_router(state: SubgraphState):
    """
    Determines whether the workflow retries
    the historical data retrieval process.
    """

    if state.get("retry_required"):
        return "call_model"

    return END
