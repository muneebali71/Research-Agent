from __future__ import annotations
 
import sys
import os
from fastmcp import FastMCP
import json
from rich import print
from tavily import TavilyClient
from langchain.tools import tool
 
import asyncio
import aiohttp
from bs4 import BeautifulSoup
# Works whether launched via: fastmcp run, python -m, or python directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ──────────────────────────────────────────────────────────────────────────────
from config import TAVILY_API_KEY






 

tavily = TavilyClient(api_key = TAVILY_API_KEY)  # Initialize Tavily client

mcp = FastMCP("WebSearchServer")

# @mcp.tool
@mcp.tool
async def web_search(query: str) -> str:
    """
    Search the web for the recent and reliable information about the query and return Titles, URLs and snippets.
    """
    response = await asyncio.to_thread(lambda:tavily.search(query = query, max_results = 5))  # Perform web search using Tavily
    out = []
    for r in response["results"]:
        out.append(
            # f"Title : {r["title"]}\n URL: {r["url"]}\n Content: {r["content"][:300]}"
            {
                "title": r["title"],
                "url": r["url"],
                "content": r["content"][:300]
            }
        )
    # return "\n---\n".join(out)
    return json.dumps(out, indent=2)



# web_search.invoke("What are the recent news of war?")
# print(asyncio.run(web_search.ainvoke("What are the recent news of war?")))

@mcp.tool
async def scrap_url(url: str) -> str:
    """
    Scrap and return clean text content from the given url for deeper reading.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url,timeout=8, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script",
    "style",
    "nav",
    "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:6000]  # Return first 3000 characters of clean text
    except Exception as e:
        return f"Error scraping URL: {str(e)}"


@mcp.resource("info://server")
async def server_info():
    """Get server information
    """
    info ={
        "name": "Web Search Server",
        "version": "1.0.0",
        "description": "Provides web search and webpage scraping for research agents.",
        "tools":["web_search","scrap_url"],
        "author": "Muneeb"
    }
    return json.dumps(info, indent=2)
# print(asyncio.run(scrap_url.ainvoke("https://www.nbcnews.com/world/iran-war")))



if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0",port = 8010)


