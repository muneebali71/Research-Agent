from __future__ import annotations
import re

import sys, os, json, asyncio, ssl
import aiohttp
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from tavily import TavilyClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import TAVILY_API_KEY

tavily = TavilyClient(api_key=TAVILY_API_KEY)
mcp    = FastMCP("WebSearchServer")

# SSL context that skips certificate verification — fixes errors on
# corporate/school networks.
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode    = ssl.CERT_NONE

# Tag/index page patterns — these are useless to scrape
_INDEX_PATTERNS = [
    "/tag/", "/tags/", "/topic/", "/topics/", "/category/",
    "/search?", "/search/", "?q=", "#",
]


def _is_index_url(url: str) -> bool:
    """Return True if this URL looks like a tag/topic/search index page."""
    u = url.lower()
    return any(p in u for p in _INDEX_PATTERNS)


@mcp.tool
async def web_search(query: str) -> str:
    """Search the web and return titles, URLs and full content snippets.

    Uses Tavily advanced search depth for real article URLs and longer
    content (up to 800 chars per result). Filters out tag/index pages.
    Appends recency keywords when the query has no time scope.
    """
    # Append "latest news 2026" if not already time-scoped — nudges Tavily
    # toward recent articles rather than evergreen/index pages.
    q = query.strip()
    if not any(yr in q.lower() for yr in ["2024", "2025", "2026", "latest", "recent"]):
        q = q + " latest news 2026"

    response = await asyncio.to_thread(
        lambda: tavily.search(
            query=q,
            max_results=7,
            search_depth="advanced",
            include_answer=False,
        )
    )
    results = response.get("results", [])

    # Filter out tag/index pages — they contain no article facts
    real_articles = [r for r in results if not _is_index_url(r.get("url", ""))]
    if not real_articles:
        real_articles = results  # fallback: use all if everything filtered out

    out = []
    for r in real_articles[:5]:
        out.append({
            "title":          r.get("title", "Untitled"),
            "url":            r.get("url", ""),
            "content":        r.get("content", "")[:800],
            "published_date": r.get("published_date", ""),
        })

    return json.dumps(out, indent=2)


@mcp.tool
async def scrap_url(url: str) -> str:
    """Scrape and return clean text content from a URL.

    Returns up to 4000 characters of clean article text.
    Skips tag/index pages and returns an informative error instead.
    """
    if _is_index_url(url):
        return (
            f"Skipped: '{url}' is a tag/topic index page with no article content. "
            "Please scrape a specific article URL instead."
        )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=12),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                ssl=_ssl_ctx,
            ) as resp:
                if resp.status >= 400:
                    return f"HTTP {resp.status} error fetching {url}"
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Try to find the main article body first for cleaner extraction
        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"article|content|story|body", re.I))
            or soup
        )

        text = article.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        import re as _re
        text = _re.sub(r'\n{3,}', '\n\n', text)

        return text[:4000]

    except asyncio.TimeoutError:
        return f"Timeout scraping {url} — site took too long to respond."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@mcp.resource("info://server")
async def server_info():
    return json.dumps({
        "name":    "Web Search Server",
        "version": "2.0.0",
        "tools":   ["web_search", "scrap_url"],
        "changes": [
            "web_search: uses search_depth=advanced, filters index pages, 800-char snippets",
            "scrap_url: skips index pages, extracts article body, 4000-char limit",
        ],
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8010)