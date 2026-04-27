# ⚾ Baseball IQ

**An AI-powered pre-game briefing platform** that combines Statcast data, injury reports, 
umpire tendencies, fan sentiment, and creator takes — then uses RAG + LLMs to generate 
sharp pre-game analysis for any MLB matchup.

---

## 🗂️ Project Structure

```
baseball-iq/
├── scraper/
│   ├── mlb_api.py          ← lineups, injuries, probable starters (free MLB API)
│   ├── statcast.py         ← pitch data, batter heatmaps (pybaseball)
│   ├── umpire.py           ← umpire accuracy scraper (umpscorecards.com)
│   └── reddit_youtube.py   ← fan sentiment (PRAW + YouTube Data API)
├── pipeline/
│   ├── ingest.py           ← clean + chunk all scraped data
│   ├── embed.py            ← embed chunks into ChromaDB
│   └── rag.py              ← RAG query engine (LangChain)
├── models/
│   └── predict.py          ← XGBoost matchup predictor (Track B)
├── app/
│   └── streamlit_app.py    ← dashboard UI
└── data/
    └── chroma_db/          ← local vector store

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run for today's games
```bash
python main.py
```

### 4. Run for a specific matchup
```bash
python main.py --game "Red Sox" --date 2026-04-08
```

### 5. Ask a custom question
```bash
python main.py --query "How does Rafael Devers hit against sliders?"
```

### 6. Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 How the RAG Pipeline Works

```
Raw Data (MLB API + Statcast + Umpire + Reddit + YouTube)
           │
           ▼
    Natural Language Documents
    "Freddy Peralta throws a 95.4 mph fastball 42% of the time with 29.3% whiff rate..."
           │
           ▼
    Chunking + Embedding (sentence-transformers: all-MiniLM-L6-v2)
           │
           ▼
    ChromaDB (local vector store, no server needed)
           │
           ▼
    Query: "pre-game analysis Red Sox vs Brewers"
           │
           ▼
    Retrieve top-K relevant chunks by cosine similarity
           │
           ▼
    LLM (Claude / GPT-4) generates structured pre-game brief
```

### Why RAG?
- The LLM doesn't know today's injury report, tonight's umpire, or last week's form
- RAG grounds the LLM in real, current, game-specific data
- The model generates the *narrative* — your scrapers provide the *facts*

---

## 📡 Data Sources (All Free)

| Source | What it provides | Method |
|--------|-----------------|--------|
| `statsapi.mlb.com` | Lineups, probable pitchers, injuries, game schedule | REST API, no key |
| `baseball-savant.mlb.com` | Statcast pitch data, batter heatmaps | via `pybaseball` |
| `fangraphs.com` | Advanced metrics, WAR, FIP, wRC+ | via `pybaseball` |
| `umpscorecards.com` | Umpire accuracy %, zone tendencies | BeautifulSoup scrape |
| `reddit.com` | Fan posts, injury rumors | PRAW (free dev account) |
| YouTube Data API | Creator videos, auto-transcripts | Free tier: 10k units/day |

---

## 📊 Pre-Game Brief Output Structure

```
1. Matchup Overview      — Quick snapshot, venue, weather
2. Starting Pitchers     — Arsenal, ERA, WHIP, recent trends
3. Lineup Analysis       — Hot/cold hitters, matchup advantages  
4. Injury Report         — Who's in/out, what it means
5. Umpire Card           — Accuracy %, zone tendencies tonight
6. The Hidden Story      — The angle most fans will miss
7. Prediction            — Win probability with reasoning
```

---

## 🎓 Class Project Notes (Track A — Due End of April)

The core deliverable for your LLM class:

1. **RAG Pipeline** (`pipeline/rag.py`) — This is the academic contribution:
   - Document ingestion from heterogeneous sources
   - Chunking strategy for baseball data
   - Embedding with sentence-transformers
   - ChromaDB as the vector store
   - LLM-generated structured output

2. **Demo script** — Run with mock data (no API keys needed):
   ```bash
   python pipeline/rag.py
   ```

3. **Key talking points for class**:
   - Why RAG over fine-tuning here? *Real-time data that changes daily*
   - Chunking strategy: *each data type is its own document with metadata*
   - Retrieval filter: *by game_id for precision, open for exploration*
   - Embedding model choice: *all-MiniLM-L6-v2 — fast, good, free*

---

## 🔬 Track B — Personal Extensions (Post-April)

### Matchup Prediction Model
```python
# models/predict.py
# XGBoost trained on historical Statcast data
# Features: pitcher arsenal vs batter tendencies, park factors, bullpen state
# Target: win probability, run total, strikeout over/under
```

### Fine-Tuning (Stretch Goal)
- Dataset: FanGraphs articles + Baseball Prospectus + YouTube transcripts
- Model: Mistral 7B or LLaMA 3 via Hugging Face
- Goal: baseball-native language model that sounds like a real analyst

### Rumor Classifier
```python
# NLP model to label social posts:
# "confirmed_injury" | "unverified_rumor" | "fan_speculation" | "satire"
# Model: fine-tuned DistilBERT on labeled baseball posts
```

### Creator Aggregator
- Pull Jomboy, Foolish Baseball, Pitching Ninja, Talkin' Baseball
- Extract stance on tonight's game using LLM
- Surface as "Creator Pulse" card in UI

---

## ⚙️ Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...       # Required for LLM generation
REDDIT_CLIENT_ID=...               # Optional: get free at reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=BaseballIQ/1.0
YOUTUBE_API_KEY=...                # Optional: free at console.cloud.google.com
```

---

## 🗺️ Roadmap

- [x] MLB Stats API scraper (games, lineups, injuries, pitchers)
- [x] Statcast scraper (arsenal, batter profiles, H2H)  
- [x] Umpire scraper (UmpScorecards.com)
- [x] RAG pipeline (ingest → embed → ChromaDB → LLM)
- [x] Main orchestrator
- [ ] Reddit + YouTube scrapers
- [ ] Streamlit dashboard
- [ ] XGBoost matchup predictor
- [ ] Fine-tuned narration model
- [ ] Rumor classifier
- [ ] Umpire card UI with zone visualization