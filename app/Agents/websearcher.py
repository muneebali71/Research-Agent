from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from app.llm_model.llm import get_llm
from langsmith import traceable

@traceable(name = "build search agent")
async def build_search_agent():
        client = MultiServerMCPClient({
            "websearch": {
                "transport": "streamable_http",
                "url": "http://localhost:8010/mcp",
            }
        })
        tools = await client.get_tools()
        # Only give search agent the web_search tool
        search_tools = [t for t in tools if t.name == "web_search"]
        return create_agent(model =get_llm(), tools = search_tools)
