# from app.pipline.websearch_pipline import search_node,reader_node,writer_node,critic_node,revise_node,should_revise
# from langgraph.graph import StateGraph, END
# from app.state.schema import ResearchState




# # ── Build the graph ────────────────────────────────────────────────────────────
# def build_research_graph() -> StateGraph:
#     graph = StateGraph(ResearchState)

#     # Register nodes
#     graph.add_node("search_node", search_node)
#     graph.add_node("reader_node", reader_node)
#     graph.add_node("writer_node", writer_node)
#     graph.add_node("critic_node", critic_node)
#     graph.add_node("revise_node", revise_node)

#     # Linear edges
#     graph.set_entry_point("search_node")
#     graph.add_edge("search_node", "reader_node")
#     graph.add_edge("reader_node", "writer_node")
#     graph.add_edge("writer_node", "critic_node")

#     # Conditional edge: critic → revise or END
#     graph.add_conditional_edges(
#         "critic_node",
#         should_revise,
#         {
#             "revise": "revise_node",
#             "end":    END,
#         }
#     )

#     # Revision loops back to critic
#     graph.add_edge("revise_node", "critic_node")

#     return graph.compile()



# app = build_research_graph()

# print(app)














from langgraph.graph import StateGraph, END

from app.state.schema import ResearchState
from app.langgraph_nodes.websearch_nodes import (
    search_node, reader_node, writer_node,
    critic_node, revise_node, should_revise,
)


def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search_node", search_node)
    graph.add_node("reader_node", reader_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("revise_node", revise_node)

    graph.set_entry_point("search_node")
    graph.add_edge("search_node", "reader_node")
    graph.add_edge("reader_node", "writer_node")
    graph.add_edge("writer_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        should_revise,
        {"revise": "revise_node", "end": END},
    )
    graph.add_edge("revise_node", "critic_node")

    return graph.compile()


async def run_research_agent(topic: str) -> dict:
    app = build_research_graph()

    initial_state: ResearchState = {
        "topic":             topic,
        "search_results":    "",
        "scraped_content":   "",
        "research_combined": "",
        "report":            "",
        "feedback":          "",
        "final_score":       0.0,
        "revision_count":    0,
        "revision_history":  [],
    }

    final_state = await app.ainvoke(initial_state)

    print(f"\nDone. Final score: {final_state['final_score']}/10 | Revisions: {final_state['revision_count']}")
    return final_state









# ── Visualization ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     app = build_research_graph()
 
#     # Option 1: ASCII print (always works, no dependencies)
#     app.get_graph().print_ascii()
 
#     # Option 2: Save as PNG  (requires: pip install grandalf)
#     try:
#         png_bytes = app.get_graph().draw_mermaid_png()
#         with open("graph.png", "wb") as f:
#             f.write(png_bytes)
#         print("\n✓ Graph saved to graph.png")
#     except Exception as e:
#         print(f"\n⚠ PNG export failed: {e}")
#         print("  Run: pip install grandalf")
