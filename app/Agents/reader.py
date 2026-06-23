from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


from app.llm_model.llm import get_llm
from langsmith import traceable


@traceable(name = "build reader agent")
async def build_reader_agent():
        client = MultiServerMCPClient({
            "websearch": {
                "transport": "streamable_http",
                "url": "http://localhost:8010/mcp",
            }
        }) 
        tools = await client.get_tools()
        reader_tools = [t for t in tools if t.name == "scrap_url"]
        return create_agent(model = get_llm(),tools = reader_tools)

