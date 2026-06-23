# Multi-Agent Research & Report Generation Chatbot

A multi-agent research assistant built on **LangGraph**. Every user query is routed by a **supervisor** agent into one of three research strategies — **web search**, **PDF retrieval (RAG)**, or **both combined** — after which a **writer** agent drafts a response and a **critic/revise** loop iteratively improves it until it clears a quality bar. The system is exposed as a FastAPI backend with a Streamlit chat interface.

> **Note on the tech stack:** the LLM powering every agent is **GROQ** (`meta-llama/llama-4-scout-17b-16e-instruct` via `langchain-groq`) — not Anthropic/Claude or OpenAI. This is intentional; treat this README and the source code as the source of truth over any conflicting assumptions.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How a request flows through the system](#how-a-request-flows-through-the-system)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Design notes & key implementation details](#design-notes--key-implementation-details)
- [Known limitations & gotchas](#known-limitations--gotchas)
- [Roadmap](#roadmap)

---

## Overview

Knowledge workers spend a large share of their time manually searching, reading, and synthesizing information before they can write anything useful. This system automates that loop with a small set of cooperating agents:

1. **Supervisor** — reads the query and the conversation's context (is a PDF uploaded? what's it about?) and decides how the query should be answered.
2. **RAG agent** — retrieves the most relevant chunks from an uploaded PDF, scoped to the current conversation.
3. **Search agent** — queries the web (Tavily) for current information.
4. **Reader agent** — scrapes the full text of the most relevant, real (non-index) articles returned by search, in parallel.
5. **Writer** — drafts a response, automatically choosing the right format (direct answer, structured briefing, comparison, or list) for the query.
6. **Critic** — scores the draft out of 10 on format fit, source grounding, clarity, and citation discipline.
7. **Reviser** — rewrites the draft using the critic's feedback; this loops until the score passes or a retry budget is exhausted.

The result is a single chat interface that can hold a conversation, answer from an uploaded document, research the live web, or blend both — automatically, without the user choosing a mode.

---

## Architecture

![LangGraph flow diagram](graph.png)

The diagram above is generated directly from the compiled LangGraph `StateGraph` (`app/graph/graph.py`) and reflects the actual runtime topology, not an idealized version of it.

### Node summary

| Node | File | Responsibility |
|---|---|---|
| `supervisor` | `app/Agents/supervisor.py` | Classifies the query as `chat`, `pdf`, `web`, or `both`. Skips the LLM call entirely and routes to `web` when the thread has no indexed PDF. |
| `chat_node` | `app/langgraph_nodes/websearch_nodes.py` | Handles greetings / small talk directly. No retrieval, no critic pass. |
| `rag_node` | `app/langgraph_nodes/rag_nodes.py` | Retrieves the top-k chunks from Qdrant for the current `thread_id`. |
| `search_node` | `app/langgraph_nodes/websearch_nodes.py` | Calls the `web_search` MCP tool (Tavily, advanced depth) and filters out tag/index pages. |
| `reader_node` | `app/langgraph_nodes/websearch_nodes.py` | Scrapes the top 2–3 real article URLs in parallel via `aiohttp` + `BeautifulSoup`. |
| `writer_node` | `app/chains/writer_chain.py` | Drafts the response, selecting a format based on the type of query and grounding every claim in retrieved context. |
| `critic_node` | `app/chains/critic_chain.py` | Scores the draft (`Score: N/10`, regex-parsed) on format fit, source grounding, clarity, and citation discipline. |
| `revise_node` | `app/chains/writer_chain.py` | Rewrites the draft using critic feedback. |

### Routing logic

- **No PDF uploaded** → supervisor skips the classifier LLM call entirely and routes straight to `web` (or `chat` for greetings).
- **PDF uploaded** → an LLM compares the query against a short topic hint built from the PDF's first chunks, choosing `pdf` (on-topic), `web` (off-topic), or `both` (on-topic, but needs current information).
- **`both`** flows through `rag_node` first, then continues into the web branch (`search_node → reader_node`) before reaching the writer.
- **Critic loop** repeats up to `MAX_RETRIES` (default 3) times, exiting early once the score reaches `PASS_SCORE` (default 8/10).

---

## How a request flows through the system

1. User sends a message from the Streamlit UI (or directly via the API).
2. `POST /sessions/{sid}/chat` persists the user message to Postgres and invokes the compiled LangGraph graph with `thread_id = sid`.
3. The graph runs end-to-end (`supervisor → … → writer → critic ⇄ revise → END`), using:
   - **Qdrant** (filtered by `thread_id`) for PDF retrieval, if relevant.
   - **Tavily + BeautifulSoup** for web search and scraping, if relevant.
   - **GROQ** for every LLM call along the way (routing, writing, critiquing, revising).
4. The LangGraph **Postgres checkpointer** persists the graph's internal state under the same `thread_id`, giving the conversation memory across turns.
5. The final report, route taken, and critic score are saved as an assistant message and returned to the client.

`thread_id` is the single key that ties three independent subsystems together: it is the chat session ID in Postgres, the LangGraph checkpointer key, and the Qdrant metadata filter used to scope PDF chunks to a conversation.

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM | GROQ — `meta-llama/llama-4-scout-17b-16e-instruct` (via `langchain-groq`) |
| Tool serving | FastMCP (`web_search`, `scrap_url` over streamable HTTP) |
| Web search | Tavily API |
| Web scraping | `aiohttp` + `BeautifulSoup` |
| Vector store | Qdrant (Docker) |
| Embeddings | Local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim, no API cost) |
| Relational storage | PostgreSQL (local, **not** Dockerized) |
| Async DB driver | `asyncpg` via SQLAlchemy (chat tables) |
| Conversation memory | LangGraph `AsyncPostgresSaver` checkpointer via `psycopg` (separate pool) |
| API layer | FastAPI |
| Frontend | Streamlit |
| Observability | LangSmith (`@traceable` decorators throughout) |

---

## Project structure

```
app/
├── Agents/
│   ├── pdf_reader.py          # PDF ingestion (chunk + embed + index) and the RAG tool
│   ├── reader.py               # MCP client agent wired to scrap_url
│   ├── supervisor.py            # Routing logic (chat / pdf / web / both)
│   └── websearcher.py           # MCP client agent wired to web_search
│
├── api/
│   ├── app.py                   # FastAPI app factory + lifespan (DB, checkpointer, graph)
│   └── routes.py                 # /sessions, /chat, /upload, /chunks endpoints
│
├── chains/
│   ├── critic_chain.py
│   └── writer_chain.py           # writer_chain + revision_chain
│
├── db/
│   ├── database.py               # Async SQLAlchemy engine + session factory
│   ├── checkpointer.py            # Standalone checkpointer helper (used by scripts)
│   └── model.py                   # Session, Message, Document ORM models
│
├── graph/
│   └── graph.py                    # Builds and compiles the LangGraph StateGraph
│
├── langgraph_nodes/
│   ├── rag_nodes.py
│   └── websearch_nodes.py          # search / reader / writer / critic / revise / chat nodes
│
├── llm_model/
│   └── llm.py                       # GROQ client factory
│
├── mcp_servers/
│   └── websearch_server.py          # FastMCP server exposing web_search + scrap_url
│
├── rag/
│   ├── embeddings.py                 # Local MiniLM embeddings (singleton, @lru_cache)
│   └── vector_store.py                # Qdrant client — add / retrieve / scroll by thread_id
│
├── state/
│   └── schema.py                       # ResearchState TypedDict + API Pydantic models
│
└── ui/
    ├── api_client.py
    ├── components.py
    ├── state.py
    └── streamlit_app.py

config.py
main.py
docker-compose.yml      # Qdrant only — PostgreSQL runs locally, not in Docker
requirements.txt
```

---

## Getting started

This project assumes Python ≥ 3.11 and a `uv`-managed environment (`pyproject.toml` / `uv.lock`), though `requirements.txt` is also provided for a plain `pip` setup.

### 1. Clone and install

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root (see [Configuration](#configuration) below).

### 3. Start services, in order

```bash
# 1. Qdrant — Docker is used ONLY for this
docker compose up -d

# 2. MCP tool server (web_search + scrap_url) — must be running before any web/both query
python -m app.mcp_servers.websearch_server      # http://localhost:8010/mcp

# 3. FastAPI backend (chat API + checkpointer + compiled graph)
uv run python main.py                            # http://localhost:8020  (see /docs for OpenAPI)
```

PostgreSQL must be running **locally** (not in Docker) with the database referenced in `DATABASE_URL` already created.

### 4. Start the UI

```bash
streamlit run app/ui/streamlit_app.py            # talks to the backend at http://localhost:8020
```

---

## Configuration

All configuration is read from environment variables via `config.py` / `python-dotenv`. Example `.env`:

```env
# LLM + search
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# PostgreSQL — local, not Dockerized
# Both URLs must point at the SAME database, with different drivers:
#   DATABASE_URL      -> asyncpg, used by the FastAPI app for chat tables
#   DATABASE_URL_SYNC  -> psycopg, used by the LangGraph checkpointer
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/research_agent
DATABASE_URL_SYNC=postgresql://postgres:password@localhost:5432/research_agent

# Qdrant — Dockerized
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=pdf_chunks

# Embeddings — local, no API key needed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Optional — LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=research_agent
```

| Variable | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Maximum revision attempts before the report is accepted as-is |
| `PASS_SCORE` | `8` | Minimum critic score (out of 10) required to accept a draft |
| MCP server port | `8010` | Port the FastMCP `web_search` / `scrap_url` server listens on |
| Backend port | `8020` | Port the FastAPI app listens on |
| Max Tavily results | `5–7` | Number of search results requested per query |
| Max scraped chars / article | `3000–4000` | Character cap applied per scraped page |

---

## API reference

Full interactive schema is available at `http://localhost:8020/docs` once the backend is running. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions` | Create a new chat session (its `id` doubles as the LangGraph `thread_id`) |
| `GET` | `/sessions` | List all chat sessions |
| `POST` | `/sessions/{sid}/upload` | Upload and ingest a PDF (chunked, embedded, and indexed into Qdrant) |
| `GET` | `/sessions/{sid}/documents` | List documents indexed for a session |
| `GET` | `/sessions/{sid}/chunks` | Return every Qdrant chunk stored for a session (powers the Streamlit chunk viewer) |
| `POST` | `/sessions/{sid}/chat` | Send a query; runs the full LangGraph pipeline and returns the report, route, and critic score |
| `GET` | `/sessions/{sid}/messages` | Full conversation history for a session |

---

## Design notes & key implementation details

- **One `thread_id`, three subsystems.** A chat session's Postgres primary key is reused, unmodified, as the LangGraph checkpointer's thread key *and* the Qdrant metadata filter value. This keeps conversation memory, PDF scoping, and chat history all consistent without any extra mapping table.
- **Two independent Postgres connections, by design.** The async SQLAlchemy engine (`asyncpg`) owns the chat tables (`sessions`, `messages`, `documents`). The LangGraph checkpointer opens its own `psycopg` `AsyncConnectionPool` with `autocommit=True`, `prepare_threshold=0`, and `row_factory=dict_row`, configured in the FastAPI app's lifespan. Both must point at the same database.
- **Qdrant uses one shared collection**, not one per user or session. Every chunk is tagged with `thread_id` in its metadata, and all retrieval/scroll operations filter on `metadata.thread_id` (note the `metadata.` prefix — `langchain-qdrant` nests payload there). The client, vector store, and embedding model are all cached singletons.
- **Web tools are served over MCP, not imported directly.** `app/mcp_servers/websearch_server.py` is a FastMCP HTTP server exposing `web_search` (Tavily) and `scrap_url` (`aiohttp` + `BeautifulSoup`). The search and reader agents connect as MCP *clients* and each filter the available tools down to the single tool they need.
- **Critic scoring is regex-based**, not structured output — `parse_score()` greps `Score: N/10` out of the critic's free-text feedback and falls back to `0.0` (forcing a revision) if no match is found.
- **The writer adapts its format to the query.** Rather than always producing a formal report, the writer chain chooses between a direct answer, a structured/dated briefing, a comparison, or a list, depending on what the query actually calls for — and the critic is prompted to evaluate against the format that was chosen, not penalize a short factual answer for lacking report-style headers.
- **CPU-bound work is kept off the event loop.** `ingest_pdf` runs via `asyncio.to_thread`, since the rest of the app is async-first (FastAPI, `asyncpg`, `ainvoke`).

---

## Known limitations & gotchas

- **Windows (`win32`) event loop:** `psycopg`'s async connection pool cannot run on the default `ProactorEventLoop`. Any new async entry point touching the checkpointer must explicitly use a `SelectorEventLoop`, as `main.py` already does — don't rely on just setting the event loop policy.
- **MCP server must be running first.** If a `web` or `both` query hangs or errors, check that `python -m app.mcp_servers.websearch_server` is up on port 8010.
- **No PDF in the thread ⇒ supervisor never calls the router LLM.** Everything routes straight to `web` (or `chat` for greetings), which is a deliberate cost/latency optimization, not a bug.
- **No formal test suite or linter is configured.** There is no CI step beyond the manual smoke test for persistence (`uv run python -m scripts.test_rag_db`), which exercises Postgres, the checkpointer, and thread-scoped Qdrant retrieval end-to-end.

---

## Roadmap

- [ ] Multi-PDF support per session (currently scoped to one logical `thread_id`'s combined chunk set)
- [ ] Structured (non-regex) critic scoring via `with_structured_output`
- [ ] Source-level citation rendering in the Streamlit UI
- [ ] Per-session export (report → PDF/Markdown download)
- [ ] Automated test suite and CI pipeline
- [ ] Streaming responses to the frontend instead of single-shot `ainvoke`

---

