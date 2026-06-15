# from langchain.agents import create_agent
# from app.llm_model.llm import get_llm

# from app.mcp_servers.websearch_server import scrap_url



# # URl Reader agent

# def build_reader_agent():


#     return create_agent(
#         model = get_llm(),
#         tools = [scrap_url]
#     )


# app/Agents/reader.py
from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent


from app.llm_model.llm import get_llm

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

