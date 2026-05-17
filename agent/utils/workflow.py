# agent/workflow.py
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from agent.utils.state import AgentState
from agent.utils.models.llm import model
from agent.utils.tools.search import search_tool
from agent.utils.tools.rag import finance_knowledge_base
from agent.utils.prompts import wealth_agent_prompt

# 1. Define the Tools and the ToolNode
tools = [search_tool, finance_knowledge_base]
tool_node = ToolNode(tools)

# 2. Bind the tools to the model
# This tells the LLM "You have these specific functions available"
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
    if not last_message.tool_calls:
        return END
    return "tools"


# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add our two main nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the Entry Point
workflow.add_edge(START, "agent")
# Add Conditional Edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Add a normal edge back to the agent after tools are used
workflow.add_edge("tools", "agent")

app = workflow.compile()
