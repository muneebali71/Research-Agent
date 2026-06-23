"""Research graph.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.state.schema import ResearchState
from app.Agents.supervisor import (
    supervisor_node, route_after_supervisor, route_after_rag,
)
from app.langgraph_nodes.rag_nodes import rag_node
from app.langgraph_nodes.websearch_nodes import (
    search_node, reader_node, writer_node,
    critic_node, revise_node, should_revise,
    chat_node,
)


def build_research_graph(checkpointer=None):
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("chat_node", chat_node)          # NEW: conversational replies
    graph.add_node("rag_node", rag_node)
    graph.add_node("search_node", search_node)
    graph.add_node("reader_node", reader_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("revise_node", revise_node)

    # Entry → supervisor decides the route.
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "chat_node":   "chat_node",
            "rag_node":    "rag_node",
            "search_node": "search_node",
        },
    )

    # chat_node ends immediately — no research needed.
    graph.add_edge("chat_node", END)

    # After RAG: "both" continues to the web branch, "pdf" jumps to writing.
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {"search_node": "search_node", "writer_node": "writer_node"},
    )

    # Web branch.
    graph.add_edge("search_node", "reader_node")
    graph.add_edge("reader_node", "writer_node")

    # Write → critic → revise loop.
    graph.add_edge("writer_node", "critic_node")
    graph.add_conditional_edges(
        "critic_node",
        should_revise,
        {"revise": "revise_node", "end": END},
    )
    graph.add_edge("revise_node", "critic_node")

    return graph.compile(checkpointer=checkpointer)


async def run_research_agent(topic: str, thread_id: str = "default") -> dict:
    """Run the graph for one query. ``thread_id`` scopes the PDF (Qdrant) and
    the conversation memory (checkpointer)."""
    app = build_research_graph()

    initial_state: ResearchState = {
        "topic":            topic,
        "thread_id":        thread_id,
        "revision_count":   0,
        "revision_history": [],
    }

    final_state = await app.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"\nDone. Route: {final_state.get('route')} | "
          f"Final score: {final_state.get('final_score')}/10 | "
          f"Revisions: {final_state.get('revision_count')}")
    return final_state













# ── Visualization ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building graph...")
    app = build_research_graph()
    print("Graph built successfully.")
 
    # Option 1: ASCII print (always works, no dependencies)
    app.get_graph().print_ascii()
 
    # Option 2: Save as PNG  (requires: pip install grandalf)
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("\n✓ Graph saved to graph.png")
    except Exception as e:
        print(f"\n⚠ PNG export failed: {e}")
        print("  Run: pip install grandalf")