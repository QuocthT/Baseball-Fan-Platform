# ⚾ Baseball IQ

**An AI-powered pre-game briefing platform** that pulls Statcast data, umpire tendencies, ballpark weather, and Reddit fan sentiment — then uses a local LLM to generate a structured, grounded pre-game brief for any MLB matchup.

Built for CDS 593 (LLMs) as a solo final project. The core engineering challenge was preventing the LLM from hallucinating statistics. Every architectural decision — Hybrid RAG, LangGraph routing, local model — was made in service of that constraint.

---

## 🧠 How It Works

The system uses an **Agentic Hybrid RAG** architecture with two separate data stores:

- **Exact stats** (ERA, lineups, pitch arsenals) → stored in an in-memory Pandas dictionary. Never embedded. Retrieved deterministically.
- **Qualitative context** (Reddit sentiment, umpire profiles, weather, news) → chunked, embedded with `sentence-transformers`, and stored in a local ChromaDB vector store.

A **LangGraph ReAct agent** decides per query which tool to call:
- `get_exact_matchup_stats` → hits the Pandas dictionary
- `query_qualitative_data` → runs semantic search over ChromaDB

The model never generates the brief until after tool calls complete. This prevents it from falling back on training memory for facts it was supposed to retrieve.

```
Raw Data (MLB API + Statcast + UmpScorecards + Reddit + OpenWeatherMap + RSS)
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Exact Stats → Pandas Dict (no embedding)│
    │  Qualitative → ChromaDB (embedded)       │
    └─────────────────────────────────────────┘
           │
           ▼
    LangGraph ReAct Agent
    (reasons → picks tool → retrieves → generates)
           │
           ▼
    Llama 3.2 (local via Ollama) generates Markdown brief
           │
           ▼
    Streamlit Dashboard
```

---

## 📡 Data Sources

| Source | What it provides | Method |
|--------|-----------------|--------|
| `statsapi.mlb.com` | Lineups, probable pitchers, injuries | REST API, no key needed |
| `baseball-savant.mlb.com` | Statcast pitch data, arsenal | via `pybaseball` |
| `fangraphs.com` | Advanced metrics (ERA, FIP, WHIP) | via `pybaseball` |
| `umpscorecards.com` | Umpire accuracy %, zone tendencies | BeautifulSoup scrape |
| `reddit.com` | Fan sentiment, injury rumors | PRAW (free dev account) |
| OpenWeatherMap | Ballpark weather at game time | REST API |
| MLB Trade Rumors RSS | Recent news and transactions | RSS feed |

> **Note:** YouTube API integration was explored but not completed — transcripts are listed as future work. Reddit sentiment was implemented but quality is limited by subreddit coverage; a proper NLP classifier is the planned improvement.

---

## 📊 Pre-Game Brief Output

```
1. Matchup Overview      — Venue, weather, quick snapshot
2. Starting Pitchers     — Arsenal, ERA, WHIP, recent form
3. Lineup Analysis       — Hot/cold hitters, matchup edges
4. Injury Report         — Who's in/out and what it means
5. Umpire Card           — Accuracy %, zone tendencies tonight
6. Fan Sentiment         — What Reddit is saying pre-game
7. The Hidden Story      — The angle most fans will miss
```

---

## 🗂️ Project Structure

```
baseball-iq/
├── scraper/
│   ├── mlb_api.py          ← lineups, injuries, probable starters
│   ├── statcast.py         ← pitch data, batter profiles (pybaseball)
│   ├── umpire.py           ← umpire accuracy scraper
│   └── reddit.py           ← fan sentiment (PRAW)
├── pipeline/
│   ├── ingest.py           ← clean + chunk scraped data
│   ├── embed.py            ← embed chunks into ChromaDB
│   └── rag.py              ← Hybrid RAG query engine
├── agent/
│   └── langgraph_agent.py  ← ReAct agent with tool calling
├── app/
│   └── streamlit_app.py    ← Streamlit dashboard
└── data/
    └── chroma_db/          ← local ChromaDB vector store
```

---

## 🚀 Setup & Running Locally

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/baseball-iq.git
cd baseball-iq
pip install -r requirements.txt
```

### 2. Install Ollama and pull the model

Install Ollama from [ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llama3.2:1b
```

Start the Ollama server:

```bash
ollama serve
```

### 3. Expose Ollama via Cloudflare Tunnel

Because Streamlit and Ollama need to communicate over a public URL in this setup, use [Cloudflare Tunnel](https://trycloudflare.com) to expose your local Ollama instance:

```bash
cloudflared tunnel --url http://localhost:11434
```

Copy the generated URL (something like `https://your-tunnel-name.trycloudflare.com`) — you'll need it in the next step.

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=https://your-tunnel-name.trycloudflare.com   # from step 3
YOUTUBE_API_KEY=...       # optional, not fully implemented
REDDIT_CLIENT_ID=...      # free at reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=BaseballIQ/1.0
```

> **Note:** Every time you restart the Cloudflare tunnel, you get a new URL. Update `OLLAMA_BASE_URL` in your `.env` each time.

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Key Design Decisions

**Why local Llama instead of a cloud API?**
Early testing with a cloud-hosted model produced hallucinated ERAs. Running locally allowed tighter prompt constraints — the system prompt explicitly forbids the model from inventing numbers. Exact stats are injected verbatim so the model narrates, not recalls.

**Why separate ChromaDB and Pandas?**
You cannot let a vector similarity search near real statistics. Keeping exact numbers out of the embedding space entirely and retrieving them deterministically was the simplest way to prevent retrieval from corrupting numeric answers.

**Why LangGraph?**
A standard RAG chain retrieves first and generates once with no decision-making in between. LangGraph lets the model reason about the query, decide which tool fits, and route accordingly. It also makes the routing inspectable — you can see exactly which tool was called for which query, which is useful for debugging.

---

## 🔬 Future Work

- **XGBoost matchup predictor** — train on historical Statcast data to predict strikeout totals pre-game; shadow tracker infrastructure is already in place
- **Fine-tuned narration model** — fine-tune on FanGraphs and Baseball Prospectus writing so briefs sound like a real analyst, not a generic LLM
- **NLP sentiment classifier** — replace brittle Reddit scraping with a model that labels posts as confirmed injury, unverified rumor, or fan speculation
- **YouTube transcript integration** — pull and summarize pre-game takes from baseball creators

---

## 📚 References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
- LangGraph: github.com/langchain-ai/langgraph
- ChromaDB: trychroma.com
- Ollama: ollama.com
- pybaseball: github.com/jldbc/pybaseball
- UmpScorecards: umpscorecards.com

---

## 🎓 Acknowledgements

Built as a solo final project for CDS 593 (Large Language Models). Claude (Anthropic) was used as a coding and writing assistant throughout development.
