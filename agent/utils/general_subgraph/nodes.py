from langgraph.graph import END
from agent.utils.state import AgentState
from agent.utils.models.llm import model
from agent.utils.general_subgraph.tools.search import search_tool
from agent.utils.general_subgraph.tools.rag import finance_knowledge_base
from agent.utils.general_subgraph.prompts import wealth_agent_prompt
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# 1. Define the Tools and the ToolNode
tools = [search_tool, finance_knowledge_base]
tool_node = ToolNode(tools)

# 2. Bind the tools to the model
model_with_tools = model.bind_tools(tools)


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
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END

