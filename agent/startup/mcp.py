# agent/startup/mcp.py

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from agent.utils.models.llm import model

mcp_client = MultiServerMCPClient(
    {
        "mcp_server": {
            "transport": "streamable_http",
            "url": "http://localhost:24000/mcp",
        }
    }
)

tools = None
tool_node = None
hist_model_with_tools = None


async def initialize_mcp():

    global tools
    global tool_node
    global hist_model_with_tools
    tools = await mcp_client.get_tools()

    tool_node = ToolNode(tools)
    hist_model_with_tools = model.bind_tools(tools)

    print("TOOL NODE:", tool_node)
    print("--- MCP TOOLS INITIALIZED ---")

    return {
        "tools": tools,
        "tool_node": tool_node,
        "hist_model_with_tools": hist_model_with_tools
    }


