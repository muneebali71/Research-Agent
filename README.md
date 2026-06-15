# Multi-Agent Research & Report Generation System

An AI-powered research pipeline that takes a topic as input and returns a fully written, critic-reviewed, high-quality report — with zero human intervention. Built with LangGraph, LangChain, FastMCP, and Claude.

---

## What It Does

Knowledge workers spend 40–60% of their time manually searching, reading, and synthesising information. This system automates that entire pipeline using a coordinated set of specialised AI agents:

1. **Search Agent** — queries the web via Tavily and returns the most relevant, recent results
2. **Reader Agent** — picks the best URL from search results and scrapes its full content
3. **Writer Agent** — drafts a structured, detailed report from the combined research
4. **Critic Agent** — scores the report out of 10 and triggers revisions if quality is below threshold
5. **Revision Loop** — automatically rewrites the report up to 3 times until it passes the quality bar

The result: one input topic → one polished, well-researched report.

---

## Architecture

```
User Input (topic)
        │
        ▼
 ┌─────────────┐
 │ Search Agent │  ──► Tavily Web Search (via FastMCP server)
 └─────────────┘
        │
        ▼
 ┌─────────────┐
 │ Reader Agent │  ──► URL Scraper (via FastMCP server)
 └─────────────┘
        │
        ▼
 ┌─────────────┐
 │ Writer Chain │  ──► Initial Report Draft
 └─────────────┘
        │
        ▼
 ┌─────────────┐     score < 8
 │ Critic Chain │  ──────────────► Revision Chain ──► back to Critic
 └─────────────┘
        │ score ≥ 8
        ▼
  Final Report
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM | Claude (Anthropic) |
| Tool servers | FastMCP (HTTP transport) |
| Web search | Tavily API |
| Web scraping | aiohttp + BeautifulSoup |
| Async runtime | Python asyncio |
| Terminal UI | Rich |

---

## Project Structure

```
research-agent/
├── app/
│   ├── Agents/
│   │   ├── websearcher.py       # Search agent — uses web_search MCP tool
│   │   └── reader.py            # Reader agent — uses scrap_url MCP tool
│   ├── chains/
│   │   ├── writer_chain.py      # Drafts the initial report
│   │   ├── revision_chain.py    # Rewrites report based on critic feedback
│   │   └── critic_chain.py      # Scores report out of 10
│   ├── mcp_servers/
│   │   └── websearch_server.py  # FastMCP server: web_search + scrap_url tools
│   ├── pipline/
│   │   └── websearch_pipline.py # Main pipeline: wires all agents together
│   └── llm_model/
│       └── llm.py               # LLM initialisation (Claude via Anthropic)
├── test_run.py                  # Entry point for running the pipeline
├── requirements.txt
├── .env
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Start the MCP server

Open a terminal and run:

```bash
python -m app.mcp_servers.websearch_server
# Server starts at http://localhost:8010
```

### 6. Run the pipeline

Open a second terminal and run:

```bash
python -m test_run
```

---

## How the Pipeline Works

### Step 1 — Search Agent
Sends the research topic to the FastMCP `web_search` tool, which queries Tavily and returns titles, URLs, and content snippets for the top 5 results.

### Step 2 — Reader Agent
Takes the search results, picks the most relevant URL, and calls the `scrap_url` MCP tool to scrape and return clean text content from the page (up to 6,000 characters).

### Step 3 — Writer Chain
Combines the search results and scraped content into a `research_combined` context and passes it to the writer chain, which drafts a structured, detailed report.

### Step 4 — Critic + Revision Loop
The critic chain reads the report and returns a score out of 10 with detailed feedback. If the score is below 8, the revision chain rewrites the report using the feedback. This repeats up to 3 times. If the score reaches 8 or above, the loop exits early.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Maximum revision attempts before accepting the report |
| `PASS_SCORE` | `8` | Minimum score out of 10 to accept the report |
| MCP server port | `8010` | Port the FastMCP server listens on |
| Max search results | `5` | Number of Tavily results per query |
| Max scraped chars | `6000` | Character limit for scraped page content |

---

## Example Output

```
==================================================
Step 1 — Search agent is working...
==================================================
Search results: [top 5 Tavily results with titles, URLs, snippets]

==================================================
Step 2 — Reader agent is scraping top resources...
==================================================
Scraped content: [full clean text from best URL]

==================================================
Step 3 — Writer is drafting the report...
==================================================
Initial Report: [structured markdown report]

==================================================
Step 4 (attempt 1/3) — Critic is evaluating...
==================================================
Critic Feedback (Score: 7/10): [detailed feedback]

✗ Score 7/10 is below 8/10. Requesting revision 1...

Step 4 (attempt 2/3) — Critic is evaluating...
Critic Feedback (Score: 9/10): [feedback]

✓ Report passed with score 9/10. No revision needed.

Done. Final score: 9/10
```

---

## API Keys

| Service | Free Tier | Get Key |
|---|---|---|
| Anthropic (Claude) | $5 free credit | https://console.anthropic.com |
| Tavily | 1,000 free searches/month | https://tavily.com |

---

## Roadmap

- [ ] FastAPI backend with REST endpoints
- [ ] Streamlit UI for browser-based interaction
- [ ] PostgreSQL session persistence (save research history)
- [ ] PDF upload support (PDFReader agent)
- [ ] LangSmith observability (traces, costs, latency per agent)
- [ ] Citation agent (inline references + bibliography)
- [ ] Multi-topic batch processing

---

## License

MIT License — free to use, modify, and distribute.