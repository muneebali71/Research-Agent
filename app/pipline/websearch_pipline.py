
from app.Agents.websearcher import build_search_agent
from app.Agents.reader import build_reader_agent
from langchain_core.messages import ToolMessage, AIMessage

from app.chains.writer_chain import writer_chain, revision_chain
from app.chains.critic_chain import critic_chain
from app.llm_model.llm import get_llm
import asyncio
import re
from rich import print

llm = get_llm()

MAX_RETRIES = 3          # max revision attempts
PASS_SCORE  = 8          # minimum acceptable score


def parse_score(feedback: str) -> float:
    """
    Extract the numeric score from critic output like 'Score: 7/10' or 'Score: 6.5/10'.
    Returns 0.0 if no score is found so the retry loop always triggers.
    """
    match = re.search(r"Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def extract_tool_content(messages):
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                results.append(content)
            elif isinstance(content, list):
                # each item is usually {"type": "text", "text": "..."}
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        results.append(item["text"])
                    elif isinstance(item, str):
                        results.append(item)
    return "\n\n".join(results)




async def run_research_agent(topic: str) -> dict:
    state = {}

    # ── Step 1: Search agent ───────────────────────────────────────────────────
    print("\n" + " =" * 50)
    print("Step 1 — Search agent is working...")
    print("=" * 50)

    search_agent = await build_search_agent()
    search_result = await search_agent.ainvoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    # tool_data = [msg.content for msg in search_result["messages"] if isinstance(msg, ToolMessage)]
    # state["search_results"] = "\n\n".join(tool_data)
    # print(f"\nSearch results:\n{state['search_results']}")

    state["search_results"] = extract_tool_content(search_result["messages"])

    # print (search_result)
    print(f"state search result : \n{state["search_results"]}")

    # ── Step 2: Reader agent ───────────────────────────────────────────────────
    print("\n" + " =" * 50)
    print("Step 2 — Reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = await build_reader_agent()
    reader_result = await reader_agent.ainvoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:1000]}"
        )]
    })

    # tool_data = [msg.content for msg in reader_result["messages"] if isinstance(msg, ToolMessage)]
    # state["scraped_content"] = "\n\n".join(tool_data)
    state["scraped_content"] = extract_tool_content(reader_result["messages"])
    print(f"\nScraped content:\n{state['scraped_content']}")

    # Combined research used by both writer and revision chains
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["research_combined"] = research_combined

    # ── Step 3: Write initial report ──────────────────────────────────────────
    print("\n" + " =" * 50)
    print("Step 3 — Writer is drafting the report...")
    print("=" * 50)

    state["report"] = await writer_chain.ainvoke({
        "topic": topic,
        "research": research_combined
    })
    print(f"\nInitial Report:\n{state['report']}")

    # ── Step 4: Critic + revision loop ────────────────────────────────────────
    state["revision_history"] = []   # keeps track of every (score, feedback) pair

    for attempt in range(1, MAX_RETRIES + 1):
        print("\n" + " =" * 50)
        print(f"Step 4 (attempt {attempt}/{MAX_RETRIES}) — Critic is evaluating the report...")
        print("=" * 50)

        feedback = await critic_chain.ainvoke({"report": state["report"]})
        score    = parse_score(feedback)

        print(f"\nCritic Feedback (Score: {score}/10):\n{feedback}")
        state["revision_history"].append({"attempt": attempt, "score": score, "feedback": feedback})

        if score >= PASS_SCORE:
            print(f"\n[bold green]✓ Report passed with score {score}/10. No revision needed.[/bold green]")
            break

        if attempt == MAX_RETRIES:
            print(f"\n[bold yellow]⚠ Max retries reached. Keeping best report (score: {score}/10).[/bold yellow]")
            break

        # Score too low — ask the writer to revise
        print(f"\n[bold red]✗ Score {score}/10 is below {PASS_SCORE}/10. Requesting revision {attempt}...[/bold red]")

        state["report"] = await revision_chain.ainvoke({
            "topic":    topic,
            "report":   state["report"],
            "feedback": feedback,
            "research": research_combined
        })
        print(f"\nRevised Report (attempt {attempt}):\n{state['report']}")

    # Store the final feedback from the last critic run
    state["feedback"] = state["revision_history"][-1]["feedback"]
    state["final_score"] = state["revision_history"][-1]["score"]

    print("\n" + " =" * 50)
    print(f"Done. Final score: {state['final_score']}/10")
    print("=" * 50)

    return state