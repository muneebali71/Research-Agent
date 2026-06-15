# from langchain.agents import create_agent
# from app.llm_model.llm import get_llm

# from app.mcp_servers.websearch_server import web_search,scrap_url



# # llm = get_llm()

# # websearch agent

# def build_search_agent():


#     return create_agent(
#         model = get_llm(),
#         tools = [web_search]
#     )


# # def build_reader_agent():


# #     return create_agent(
# #         model = get_llm(),
# #         tools = [scrap_url]
# #     )





# app/Agents/websearcher.py
from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from app.llm_model.llm import get_llm

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
