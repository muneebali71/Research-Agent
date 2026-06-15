import asyncio
from app.graph.graph import run_research_agent

async def main():
    topic = "latest Iran Israel war updates"
    result = await run_research_agent(topic)
    print("\nFINAL RESULT:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())