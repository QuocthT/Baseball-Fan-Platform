"""
rag.py
------
The core RAG (Retrieval-Augmented Generation) pipeline for Baseball IQ.

Flow:
  1. Load stats: structured data goes into an in-memory dictionary (no embedding).
  2. Ingest: unstructured text (sentiment, umpire, injuries) → ChromaDB.
  3. Query: LLM uses Tool Calling to route stat questions to memory and narrative questions to ChromaDB.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from datetime import date
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

# LangChain Imports for Hybrid Routing
from langchain_core.tools import tool
# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "baseball_iq"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── In-Memory Structured Store ────────────────────────────────────────────────

# This holds exact stats so the LLM doesn't have to guess from vector similarities.
CURRENT_GAME_STATS = {}

def load_structured_stats(game_data: dict):
    """Load exact math/stats into memory for the LangChain Tool to access."""
    CURRENT_GAME_STATS.clear()
    
    # Store pitcher stats
    CURRENT_GAME_STATS["home_pitcher"] = game_data.get("home_pitcher_stats", {})
    CURRENT_GAME_STATS["away_pitcher"] = game_data.get("away_pitcher_stats", {})
    
    # Store pitch arsenals
    CURRENT_GAME_STATS["home_pitcher_arsenal"] = game_data.get("home_pitcher_arsenal", {})
    CURRENT_GAME_STATS["away_pitcher_arsenal"] = game_data.get("away_pitcher_arsenal", {})
    
    # Store lineups as markdown tables for the LLM to easily read
    home_lineup = game_data.get("home_lineup", {}).get("batting_order", [])
    away_lineup = game_data.get("away_lineup", {}).get("batting_order", [])
    
    if home_lineup:
        CURRENT_GAME_STATS["home_lineup"] = pd.DataFrame(home_lineup).to_markdown()
    if away_lineup:
        CURRENT_GAME_STATS["away_lineup"] = pd.DataFrame(away_lineup).to_markdown()

# ── Vector Store Setup ────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    except (ImportError, ValueError):
        ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Document Ingestion (Qualitative ONLY) ─────────────────────────────────────

def ingest_game_data(game_data: dict) -> int:
    """
    Convert ONLY qualitative/text data into documents and store in ChromaDB.
    (Stats are handled by load_structured_stats).
    """
    # First, load the stats into memory
    load_structured_stats(game_data)
    
    collection = get_collection()
    docs = []
    metadatas = []
    ids = []

    game = game_data.get("game", {})
    game_id = str(game.get("game_id", "unknown"))
    matchup = f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}"
    game_date = game.get("date", date.today().isoformat())

    def add_doc(text: str, doc_type: str, **extra_meta):
        doc_id = hashlib.md5(f"{game_id}:{doc_type}:{text[:50]}".encode()).hexdigest()
        docs.append(text)
        metadatas.append({
            "game_id": game_id,
            "matchup": matchup,
            "date": game_date,
            "type": doc_type,
            **{k: str(v) for k, v in extra_meta.items()},
        })
        ids.append(doc_id)

    # 1. Game overview
    add_doc(f"Game: {matchup} on {game_date}. Venue: {game.get('venue', 'Unknown')}.", doc_type="game_overview")

    # 2. Batter weaknesses (Drop the raw numbers, keep the text analysis)
    for profile in game_data.get("batter_profiles", []):
        if "error" in profile: continue
        weaknesses = profile.get("weakness_summary", [])
        if weaknesses:
            add_doc(
                f"Batter scouting for {profile['batter']}: " + "; ".join(weaknesses),
                doc_type="batter_scouting", player=profile["batter"]
            )

    # 3. Injuries
    injuries = game_data.get("injuries", [])
    if injuries:
        inj_lines = [f"{i['player']} ({i['team']}) — {i['type']}: {i.get('description', '')}" for i in injuries[:10]]
        add_doc("Recent injuries: " + " | ".join(inj_lines), doc_type="injuries")

    # 4. Umpire
    ump = game_data.get("umpire", {})
    if ump:
        add_doc(ump.get("narrative", f"Umpire {ump.get('umpire', 'Unknown')} is assigned."), doc_type="umpire")

    # 5. Reddit / Fan sentiment
    for post in game_data.get("reddit_posts", [])[:5]:
        add_doc(f"Reddit Buzz ({post.get('subreddit', 'r/baseball')}): {post.get('title', '')} — {post.get('body', '')[:300]}", doc_type="reddit")

    # 6. YouTube / Creator takes
    for yt in game_data.get("youtube_summaries", [])[:5]:
        add_doc(f"YouTube Analysis ({yt.get('channel', 'Unknown')}): {yt.get('summary', '')[:400]}", doc_type="youtube")
    
    # 7. Written Analyst News
    for article in game_data.get("analyst_news", []):
        add_doc(
            f"Analyst Report ({article.get('team', 'Unknown')}): {article.get('title', '')} — {article.get('summary', '')}", 
            doc_type="analyst_news"
        )

    # Clear old collection and batch upsert
    try:
        collection.delete(where={"game_id": game_id})
    except Exception:
        pass
        
    if docs:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return len(docs)


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def get_exact_matchup_stats(target: str) -> str:
    """
    Use this tool to get EXACT mathematical stats (ERA, WHIP, K/9, Batting AVG, OPS, Arsenal).
    Valid targets: 'home_pitcher', 'away_pitcher', 'home_pitcher_arsenal', 'away_pitcher_arsenal', 'home_lineup', 'away_lineup'.
    """
    data = CURRENT_GAME_STATS.get(target)
    if not data:
        return f"Exact stats for {target} are currently unavailable."
    return str(data)

@tool
def query_qualitative_data(query: str) -> str:
    """
    Use this tool to search for qualitative information: Fan sentiment, Reddit buzz, 
    YouTube commentary, injury context, batter weaknesses, and umpire tendencies.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=6)
    chunks = results.get("documents", [[]])[0]
    return "\n\n".join(chunks) if chunks else "No narrative data found in vector DB."


# ── Hybrid RAG Agent ──────────────────────────────────────────────────────────
def query_pregame_brief(matchup: str, game_id: Optional[str] = None) -> str:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage

    llm = ChatOllama(
        model="llama3.1",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1
    )

    tools = [get_exact_matchup_stats, query_qualitative_data]

    system_prompt = """You are Baseball IQ — an elite baseball analyst. 
    Write a sharp, accurate pre-game brief for the given matchup.
    
    CRITICAL RULES:
    1. For any statistical data (ERA, lineups, velocity, averages), you MUST use the `get_exact_matchup_stats` tool. DO NOT make up numbers.
    2. For narrative, fan sentiment, weaknesses, injuries, and umpire data, use the `query_qualitative_data` tool.
    
    Structure your brief clearly with Markdown headers:
    - ⚾ Matchup Overview
    - 🔥 Pitching Breakdown (Use exact stats & arsenal)
    - 👥 Lineup Analysis (Who is hot/cold)
    - 🏥 Injury Report
    - 🧑‍⚖️ Umpire Card & Fan Pulse (Combine sentiment and umpire tendencies)
    - 🔮 The Hidden Story & Prediction"""

    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=system_prompt)  # ✅ correct for langgraph 0.2+
    )

    try:
        response = agent_executor.invoke({"messages": [("user", f"Generate the detailed pre-game brief for: {matchup}")]})
        return response["messages"][-1].content
    except Exception as e:
        return f"⚠️ Agent execution error: {e}"
    
def query_custom(question: str, n_results: int = 8) -> str:
    """
    Answer any generic baseball question using the qualitative RAG knowledge base.
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count() or 1),
    )

    chunks = results.get("documents", [[]])[0]
    if not chunks:
        return "No relevant data found. Try ingesting game data first."
    
    context = "\n\n".join(chunks)
         
    llm = ChatOllama(
        model="llama3.1",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1
    )
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely using only the provided data."
    
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# ── Utilities ─────────────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Returns info about what's currently in the vector store."""
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"total_documents": 0, "message": "No data ingested yet."}

    sample = collection.get(limit=min(count, 50))
    types = {}
    matchups = set()
    for meta in sample.get("metadatas", []):
        t = meta.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
        matchups.add(meta.get("matchup", "?"))

    return {
        "total_documents": count,
        "document_types": types,
        "matchups_in_db": list(matchups),
    }


def clear_collection():
    """Wipe the collection. Useful for testing."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"✅ Cleared collection: {COLLECTION_NAME}")
    except Exception:
        print("Collection was already empty.")


# ── Demo: Ingest Mock Data + Query ────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Baseball IQ — Hybrid RAG Pipeline Demo")
    print("=" * 60)

    # Mock game data (replaces real scraped data for testing)
    mock_game_data = {
        "game": {
            "game_id": "999001",
            "date": "2026-04-08",
            "away_team": "Boston Red Sox",
            "home_team": "Milwaukee Brewers",
            "venue": "American Family Field",
        },
        "home_pitcher_stats": {
            "name": "Freddy Peralta",
            "era": "3.21",
            "whip": "1.08",
            "strikeouts": 187,
            "innings_pitched": "168.2",
            "wins": 12,
            "losses": 7,
            "strikeout_per_9": "9.98",
        },
        "away_pitcher_stats": {
            "name": "Brayan Bello",
            "era": "3.89",
            "whip": "1.22",
            "strikeouts": 143,
            "innings_pitched": "152.1",
            "wins": 9,
            "losses": 9,
            "strikeout_per_9": "8.44",
        },
        "home_pitcher_arsenal": {
            "pitcher": "Freddy Peralta",
            "arsenal": {
                "FF": {"usage_pct": 42.1, "avg_velocity": 95.4, "whiff_rate": 29.3},
                "SL": {"usage_pct": 33.8, "avg_velocity": 85.2, "whiff_rate": 38.1},
                "CH": {"usage_pct": 24.1, "avg_velocity": 86.1, "whiff_rate": 32.7},
            }
        },
        "away_pitcher_arsenal": {
            "pitcher": "Brayan Bello",
            "arsenal": {
                "SI": {"usage_pct": 48.3, "avg_velocity": 93.7, "whiff_rate": 18.2},
                "SL": {"usage_pct": 29.4, "avg_velocity": 83.9, "whiff_rate": 31.4},
                "CH": {"usage_pct": 22.3, "avg_velocity": 84.2, "whiff_rate": 27.8},
            }
        },
        "batter_profiles": [
            {
                "batter": "Rafael Devers",
                "weakness_summary": [
                    "Struggles vs SL — 34.2% whiff rate on 89 pitches seen",
                ],
            },
        ],
        "injuries": [
            {
                "player": "Trevor Story",
                "team": "Boston Red Sox",
                "type": "10-Day IL",
                "date": "2026-04-05",
                "description": "Right elbow inflammation",
            },
        ],
        "umpire": {
            "umpire": "Dan Bellino",
            "narrative": (
                "Dan Bellino is above-average in accuracy at 93.1%. "
                "He runs a tight zone on the outer half but is consistent. "
            ),
        },
        "reddit_posts": [
            {
                "subreddit": "r/redsox",
                "title": "Bello needs to get out of the 2nd inning habit",
                "body": "He's given up the most runs in the 2nd inning of any starter this year.",
            },
        ],
        "youtube_summaries": [
            {
                "channel": "Jomboy Media",
                "title": "Brewers are sneaky good in April",
                "summary": "Milwaukee's bullpen has a 2.14 ERA in April. They're a nightmare late-game opponent.",
            },
        ],
    }

    # Ingest
    print("\n📥 Ingesting mock game data...")
    n = ingest_game_data(mock_game_data)
    print(f"   Stored {n} documents in ChromaDB")
    print(f"   Loaded Structured Stats into memory.")

    # Stats
    print("\n📊 Vector store contents:")
    stats = get_collection_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Query
    print("\n🔍 Generating AI Agent pre-game brief...")
    print("-" * 60)
    brief = query_pregame_brief("Boston Red Sox vs Milwaukee Brewers")
    print(brief)