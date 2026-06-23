from __future__ import annotations

import re
import json
import asyncio

from langchain_core.messages import ToolMessage
from rich import print

from app.Agents.websearcher import build_search_agent
from app.chains.writer_chain import writer_chain, revision_chain
from app.chains.critic_chain import critic_chain
from app.llm_model.llm import get_llm
from app.state.schema import ResearchState

MAX_RETRIES = 3
PASS_SCORE  = 8


_INDEX_PATTERNS = [
    "/tag/", "/tags/", "/topic/", "/topics/", "/category/",
    "/search?", "/search/", "?q=", "#",
]


# __________________ Helper___________________________________
def parse_score(feedback: str) -> float:
    match = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def extract_tool_content(messages) -> str:
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                results.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        results.append(item["text"])
                    elif isinstance(item, str):
                        results.append(item)
    return "\n\n".join(results)


def parse_search_results(search_results: str) -> list[dict]:
    """Parse Tavily JSON into list of {title, url, content, published_date}."""
    try:
        data = json.loads(search_results)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def is_index_url(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in _INDEX_PATTERNS)


def get_scrapable_urls(results: list[dict], max_urls: int = 3) -> list[dict]:
    """Return up to max_urls results whose URLs point to real articles,
    preferring credible news outlets over blogs or YouTube."""
    # Preferred domains (partial match)
    preferred = [
        "reuters.com", "apnews.com", "bbc.com", "aljazeera.com",
        "theguardian.com", "nytimes.com", "washingtonpost.com",
        "cbsnews.com", "nbcnews.com", "sky.com", "france24.com",
        "dw.com", "euronews.com", "axios.com", "politico.com",
        "commonslibrary.parliament.uk",
    ]
    # Domains to skip
    skip = ["youtube.com", "youtu.be", "reddit.com", "twitter.com",
            "x.com", "facebook.com", "instagram.com"]

    # Filter out index pages and social/video sites
    candidates = [
        r for r in results
        if r.get("url")
        and not is_index_url(r["url"])
        and not any(s in r["url"].lower() for s in skip)
    ]

    # Sort: preferred domains first
    def rank(r):
        url = r.get("url", "").lower()
        for i, domain in enumerate(preferred):
            if domain in url:
                return i
        return len(preferred)

    candidates.sort(key=rank)
    return candidates[:max_urls]


async def scrape_url_direct(url: str, timeout: int = 10) -> str:
    """Scrape a URL directly via aiohttp + BeautifulSoup without going
    through the MCP agent. Faster and more reliable for multi-URL scraping."""
    import ssl as _ssl
    import aiohttp
    from bs4 import BeautifulSoup

    ssl_ctx = _ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = _ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                ssl=ssl_ctx,
            ) as resp:
                if resp.status >= 400:
                    return f"[HTTP {resp.status}]"
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Prefer article/main body
        body = (
            soup.find("article")
            or soup.find("main")
            or soup
        )
        text = body.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text[:3000]

    except asyncio.TimeoutError:
        return "[Timeout]"
    except Exception as e:
        return f"[Error: {e}]"


def build_sources(state: ResearchState) -> str:
    """Build a detailed sources block with title + URL + snippet per result."""
    parts = []

    # PDF chunks
    if state.get("pdf_chunks_meta"):
        parts.append("PDF Sources:")
        seen = set()
        for meta in state["pdf_chunks_meta"]:
            fname = meta.get("source_file", "uploaded PDF")
            page  = meta.get("page")
            key   = (fname, page)
            if key not in seen:
                seen.add(key)
                page_str = f" p.{page + 1}" if page is not None else ""
                parts.append(f"  - {fname}{page_str}")

    # Web sources — title + URL + snippet so writer can attribute accurately
    if state.get("search_results"):
        results = parse_search_results(state["search_results"])
        real = [r for r in results if not is_index_url(r.get("url", ""))]
        if real:
            parts.append("Web Sources:")
            for r in real:
                title   = r.get("title", "Untitled")
                url     = r.get("url", "")
                snippet = r.get("content", "")[:300].replace("\n", " ")
                date    = r.get("published_date", "")
                date_str = f" [{date}]" if date else ""
                parts.append(
                    f'  - {title}{date_str}\n'
                    f'    URL: {url}\n'
                    f'    Snippet: "{snippet}"'
                )

    return "\n".join(parts) if parts else "No sources available."


def _build_research(state: ResearchState) -> str:
    """Assemble full research text from whichever branches ran."""
    parts = []

    if state.get("pdf_context"):
        parts.append(f"PDF CONTENT:\n{state['pdf_context']}")

    if state.get("search_results"):
        results = parse_search_results(state["search_results"])
        real = [r for r in results if not is_index_url(r.get("url", ""))]
        if real:
            formatted = []
            for r in real:
                date = r.get("published_date", "")
                date_str = f" (published: {date})" if date else ""
                formatted.append(
                    f"SOURCE: [{r.get('title','Untitled')}]{date_str}\n"
                    f"URL: {r.get('url','')}\n"
                    f"CONTENT: {r.get('content','')}"
                )
            parts.append("SEARCH RESULTS:\n" + "\n\n---\n\n".join(formatted))
        else:
            # All results were index pages — still include them with a warning
            parts.append(
                "SEARCH RESULTS (note: these are index/tag pages with limited content):\n"
                + state["search_results"]
            )

    if state.get("scraped_content"):
        parts.append(f"SCRAPED ARTICLE CONTENT:\n{state['scraped_content']}")

    return "\n\n".join(parts)


# __________________ Chat node ___________________________________
async def chat_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: chat_node — answering conversationally...")
    print("=" * 50)
    llm = get_llm()
    response = await llm.ainvoke([
        ("system",
         "You are a helpful research assistant chatbot. "
         "Respond naturally and helpfully to the user's conversational message. "
         "If they ask what you can do, explain that you can: research topics via "
         "web search, answer questions about uploaded PDFs, or combine both. "
         "Keep your reply concise and friendly."),
        ("human", state["topic"]),
    ])
    answer = response.content if hasattr(response, "content") else str(response)
    return {"report": answer, "final_score": None}


# __________________ Search node ___________________________________
async def search_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: search_node — Search agent working...")
    print("=" * 50)
    agent = await build_search_agent()
    topic = state["topic"]

    search_prompt = f"Search for the latest 2026 news about: {topic}"

    result = await agent.ainvoke({
        "messages": [("user", search_prompt)]
    })
    search_results = extract_tool_content(result["messages"])
    print(f"Search results (preview):\n{search_results[:500]}...")
    return {"search_results": search_results}


# __________________ Reader node ___________________________________
async def reader_node(state: ResearchState) -> dict:
    """Scrape the top 2-3 real article URLs in parallel for richer content.

    This replaces the single-URL agent approach. Scraping multiple articles
    directly gives the writer far more verified facts to work with, and
    avoids the agent picking index/tag pages.
    """
    print("\n" + "=" * 50)
    print("Node: reader_node — Scraping top articles...")
    print("=" * 50)

    results = parse_search_results(state.get("search_results", ""))
    to_scrape = get_scrapable_urls(results, max_urls=3)

    if not to_scrape:
        print("[reader_node] No scrapable URLs found — skipping.")
        return {"scraped_content": ""}

    print(f"[reader_node] Scraping {len(to_scrape)} URLs in parallel:")
    for r in to_scrape:
        print(f"  → {r['url']}")

    # Scrape in parallel with asyncio.gather
    scrape_tasks = [scrape_url_direct(r["url"]) for r in to_scrape]
    scraped_texts = await asyncio.gather(*scrape_tasks)

    # Combine: label each scraped block with its source
    combined_parts = []
    for r, text in zip(to_scrape, scraped_texts):
        if text and not text.startswith("["):  # skip error/timeout results
            combined_parts.append(
                f"=== ARTICLE: {r.get('title','Untitled')} ===\n"
                f"URL: {r['url']}\n\n"
                f"{text}"
            )
            print(f"  ✅ {r['url'][:60]}... ({len(text)} chars)")
        else:
            print(f"  ⚠ {r['url'][:60]}... {text}")

    scraped_content = "\n\n" + ("=" * 60) + "\n\n".join(combined_parts)
    print(f"\n[reader_node] Total scraped: {len(scraped_content)} chars across {len(combined_parts)} articles")
    return {"scraped_content": scraped_content}


# __________________ writer node ___________________________________
async def writer_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: writer_node — Drafting report...")
    print("=" * 50)
    research_combined = _build_research(state)
    sources           = build_sources(state)

    print(f"[writer] Research size: {len(research_combined)} chars")
    print(f"[writer] Sources:\n{sources[:600]}\n")

    report = await writer_chain.ainvoke({
        "topic":    state["topic"],
        "research": research_combined,
        "sources":  sources,
    })
    print(f"Report (preview):\n{report[:300]}...")
    return {"report": report, "research_combined": research_combined, "sources": sources}


# __________________ Critic node ___________________________________
async def critic_node(state: ResearchState) -> dict:
    attempt = state.get("revision_count", 0) + 1
    print("\n" + "=" * 50)
    print(f"Node: critic_node (attempt {attempt}/{MAX_RETRIES})")
    print("=" * 50)


    sources_context = state.get("sources", "") or state.get("research_combined", "") or "No sources available."

    feedback = await critic_chain.ainvoke({
        "topic":   state["topic"],
        "report":  state["report"],
        "sources": sources_context,
    })
    score    = parse_score(feedback)
    print(f"Score: {score}/10\n{feedback}")
    return {
        "feedback":         feedback,
        "final_score":      score,
        "revision_history": [{"attempt": attempt, "score": score, "feedback": feedback}],
    }


# ── Revise node ────────────────────────────────────────────────────────────────
async def revise_node(state: ResearchState) -> dict:
    count = state.get("revision_count", 0) + 1
    print("\n" + "=" * 50)
    print(f"Node: revise_node — Revision #{count}")
    print("=" * 50)
    revised = await revision_chain.ainvoke({
        "topic":    state["topic"],
        "report":   state["report"],
        "feedback": state["feedback"],
        "research": state["research_combined"],
        "sources":  state.get("sources", "No sources available."),
    })
    print(f"Revised (preview):\n{revised[:200]}...")
    return {"report": revised, "revision_count": count}


# ── Conditional edge ───────────────────────────────────────────────────────────
def should_revise(state: ResearchState) -> str:
    score = state.get("final_score", 0.0)
    revision_count = state.get("revision_count", 0)
    if score >= PASS_SCORE:
        print(f"\n[bold green]✓ Score {score}/10 ≥ {PASS_SCORE}. Accepted![/bold green]")
        return "end"
    if revision_count >= MAX_RETRIES:
        print(f"\n[bold yellow]⚠ Max retries. Score: {score}/10[/bold yellow]")
        return "end"
    print(f"\n[bold red]✗ Score {score}/10 < {PASS_SCORE}. Revision #{revision_count+1}[/bold red]")
    return "revise"