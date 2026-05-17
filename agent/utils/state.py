from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state object that dictates what memory the agent has access to.
    Every node in the graph will read from and write to this dictionary.
    """

    # 'add_messages' is a built-in reducer. Instead of overwriting the messages list
    # every time a node returns new data, it cleanly appends the new messages.
    messages: Annotated[list[BaseMessage], add_messages]

    # A crucial addition for long-running servers:
    summary: str
