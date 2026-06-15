from __future__ import annotations

import re

from langchain_core.messages import ToolMessage
from rich import print

from app.Agents.websearcher import build_search_agent
from app.Agents.reader import build_reader_agent
from app.chains.writer_chain import writer_chain, revision_chain
from app.chains.critic_chain import critic_chain
from app.state.schema import ResearchState   # ← only schema, NOT graph

MAX_RETRIES = 3
PASS_SCORE  = 8


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Nodes ──────────────────────────────────────────────────────────────────────
async def search_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: search_node — Search agent is working...")
    print("=" * 50)
    agent  = await build_search_agent()
    result = await agent.ainvoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {state['topic']}")]
    })
    search_results = extract_tool_content(result["messages"])
    print(f"\nSearch results (preview):\n{search_results[:500]}...")
    return {"search_results": search_results}


async def reader_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: reader_node — Reader agent is scraping top resources...")
    print("=" * 50)
    agent  = await build_reader_agent()
    result = await agent.ainvoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{state['topic']}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:1000]}"
        )]
    })
    scraped_content   = extract_tool_content(result["messages"])
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{scraped_content}"
    )
    print(f"\nScraped content (preview):\n{scraped_content[:300]}...")
    return {
        "scraped_content":   scraped_content,
        "research_combined": research_combined,
    }


async def writer_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: writer_node — Writer is drafting the report...")
    print("=" * 50)
    report = await writer_chain.ainvoke({
        "topic":    state["topic"],
        "research": state["research_combined"],
    })
    print(f"\nInitial Report (preview):\n{report[:300]}...")
    return {"report": report}


async def critic_node(state: ResearchState) -> dict:
    attempt = state.get("revision_count", 0) + 1
    print("\n" + "=" * 50)
    print(f"Node: critic_node (attempt {attempt}/{MAX_RETRIES}) — Evaluating report...")
    print("=" * 50)
    feedback = await critic_chain.ainvoke({"report": state["report"]})
    score    = parse_score(feedback)
    print(f"\nCritic Feedback (Score: {score}/10):\n{feedback}")
    return {
        "feedback":         feedback,
        "final_score":      score,
        "revision_history": [{"attempt": attempt, "score": score, "feedback": feedback}],
    }


async def revise_node(state: ResearchState) -> dict:
    count = state.get("revision_count", 0) + 1
    print("\n" + "=" * 50)
    print(f"Node: revise_node — Revision #{count} in progress...")
    print("=" * 50)
    revised = await revision_chain.ainvoke({
        "topic":    state["topic"],
        "report":   state["report"],
        "feedback": state["feedback"],
        "research": state["research_combined"],
    })
    print(f"\nRevised Report (preview):\n{revised[:300]}...")
    return {
        "report":         revised,
        "revision_count": count,
    }


# ── Conditional edge function ──────────────────────────────────────────────────
def should_revise(state: ResearchState) -> str:
    score          = state.get("final_score", 0.0)
    revision_count = state.get("revision_count", 0)

    if score >= PASS_SCORE:
        print(f"\n[bold green]✓ Score {score}/10 ≥ {PASS_SCORE}. Report accepted![/bold green]")
        return "end"

    if revision_count >= MAX_RETRIES:
        print(f"\n[bold yellow]⚠ Max retries ({MAX_RETRIES}) reached. Score: {score}/10[/bold yellow]")
        return "end"

    print(f"\n[bold red]✗ Score {score}/10 < {PASS_SCORE}. Revision #{revision_count + 1}...[/bold red]")
    return "revise"