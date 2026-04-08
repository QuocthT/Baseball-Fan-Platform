"""
rag.py
------
The core RAG (Retrieval-Augmented Generation) pipeline for Baseball IQ.

Flow:
  1. Ingest: convert raw scraped data → natural language "documents"
  2. Embed:  chunk + store in ChromaDB (local vector DB, no server needed)
  3. Query:  retrieve relevant chunks → send to LLM → get pre-game brief

LLM: Claude (via Anthropic API) or any OpenAI-compatible model.
Vector DB: ChromaDB (local, free, no setup).

This is your class project core — RAG done properly.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from datetime import date

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

# LLM imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "baseball_iq"

# Uses a local sentence-transformer model for embeddings (free, no API key)
# Model downloads automatically on first run (~90MB)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Vector Store Setup ────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """
    Initialize (or load) the ChromaDB collection.
    ChromaDB stores everything locally in /data/chroma_db — no server needed.
    """
    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Use chromadb's built-in default embeddings (no extra install needed)
    # Swap for SentenceTransformerEmbeddingFunction locally for better quality
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


# ── Document Ingestion ────────────────────────────────────────────────────────

def ingest_game_data(game_data: dict) -> int:
    """
    Convert a full game data bundle into documents and store in ChromaDB.

    game_data structure:
    {
      "game": {...},              # from mlb_api.get_games()
      "home_lineup": {...},       # from mlb_api.get_lineup()
      "away_lineup": {...},
      "home_pitcher_stats": {...},# from mlb_api.get_pitcher_stats()
      "away_pitcher_stats": {...},
      "home_pitcher_arsenal": {},  # from statcast.get_pitcher_arsenal()
      "away_pitcher_arsenal": {},
      "batter_profiles": [...],   # from statcast.get_batter_profile()
      "injuries": [...],          # from mlb_api.get_injuries()
      "umpire": {...},            # from umpire.get_umpire_info()
      "reddit_posts": [...],      # from reddit_youtube.py
      "youtube_summaries": [...],
    }

    Returns: number of documents ingested
    """
    collection = get_collection()
    docs = []
    metadatas = []
    ids = []

    game = game_data.get("game", {})
    game_id = str(game.get("game_id", "unknown"))
    matchup = f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}"
    game_date = game.get("date", date.today().isoformat())

    def add_doc(text: str, doc_type: str, **extra_meta):
        """Helper to add a document with consistent metadata."""
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
    add_doc(
        f"Game: {matchup} on {game_date}. "
        f"Venue: {game.get('venue', 'Unknown')}. "
        f"Home probable pitcher: {game.get('home_probable_pitcher', {}).get('name', 'TBD')}. "
        f"Away probable pitcher: {game.get('away_probable_pitcher', {}).get('name', 'TBD')}.",
        doc_type="game_overview"
    )

    # 2. Pitcher stats
    for side in ["home", "away"]:
        ps = game_data.get(f"{side}_pitcher_stats", {})
        if ps and "era" in ps:
            add_doc(
                f"{side.capitalize()} starting pitcher {ps.get('name', 'Unknown')}: "
                f"ERA {ps.get('era')}, WHIP {ps.get('whip')}, "
                f"{ps.get('strikeouts')} strikeouts in {ps.get('innings_pitched')} IP, "
                f"W-L record {ps.get('wins')}-{ps.get('losses')}, "
                f"K/9 {ps.get('strikeout_per_9')}.",
                doc_type=f"{side}_pitcher_stats",
                pitcher=ps.get("name", "")
            )

    # 3. Pitcher arsenal
    for side in ["home", "away"]:
        arsenal_data = game_data.get(f"{side}_pitcher_arsenal", {})
        arsenal = arsenal_data.get("arsenal", {})
        if arsenal:
            pitch_lines = []
            for pitch_type, stats in arsenal.items():
                pitch_lines.append(
                    f"{pitch_type} ({stats['usage_pct']}% usage, "
                    f"{stats['avg_velocity']} mph, {stats['whiff_rate']}% whiff rate)"
                )
            add_doc(
                f"{side.capitalize()} pitcher {arsenal_data.get('pitcher', '')} arsenal: "
                + ", ".join(pitch_lines) + ".",
                doc_type=f"{side}_pitcher_arsenal",
                pitcher=arsenal_data.get("pitcher", "")
            )

    # 4. Lineups
    for side in ["home", "away"]:
        lineup_data = game_data.get(f"{side}_lineup", {})
        batters = lineup_data.get("batting_order", [])
        if batters:
            batter_lines = [
                f"{b['name']} ({b['position']}, .{str(b.get('batting_avg','.???')).replace('.','')[:3]} AVG, "
                f"{b.get('home_runs', 0)} HR)"
                for b in batters
            ]
            add_doc(
                f"{side.capitalize()} team batting order for {lineup_data.get('team', side)}: "
                + "; ".join(batter_lines) + ".",
                doc_type=f"{side}_lineup"
            )

    # 5. Batter profiles (weaknesses and strengths)
    for profile in game_data.get("batter_profiles", []):
        if "error" in profile:
            continue
        overall = profile.get("overall", {})
        weaknesses = profile.get("weakness_summary", [])
        weakness_text = (
            " Weaknesses: " + "; ".join(weaknesses) if weaknesses
            else " No significant documented weaknesses."
        )
        add_doc(
            f"Batter profile for {profile['batter']}: "
            f"Chase rate {overall.get('chase_rate')}%, "
            f"contact rate {overall.get('contact_rate')}%, "
            f"hard hit {overall.get('hard_hit_pct')}%, "
            f"avg exit velocity {overall.get('avg_exit_velocity')} mph."
            + weakness_text,
            doc_type="batter_profile",
            player=profile["batter"]
        )

    # 6. Injuries
    injuries = game_data.get("injuries", [])
    if injuries:
        inj_lines = [
            f"{i['player']} ({i['team']}) — {i['type']} on {i['date']}: {i.get('description', '')}"
            for i in injuries[:10]
        ]
        add_doc(
            "Recent injury and IL transactions: " + " | ".join(inj_lines),
            doc_type="injuries"
        )

    # 7. Umpire
    ump = game_data.get("umpire", {})
    if ump:
        add_doc(
            ump.get("narrative", f"Umpire {ump.get('umpire', 'Unknown')} is assigned."),
            doc_type="umpire",
            umpire=ump.get("umpire", "")
        )

    # 8. Reddit / fan sentiment
    for post in game_data.get("reddit_posts", [])[:5]:
        add_doc(
            f"Fan post ({post.get('subreddit', 'r/baseball')}): {post.get('title', '')} — "
            f"{post.get('body', '')[:300]}",
            doc_type="fan_sentiment",
            source="reddit"
        )

    # 9. YouTube / creator takes
    for yt in game_data.get("youtube_summaries", [])[:5]:
        add_doc(
            f"Creator take from {yt.get('channel', 'Unknown')} — '{yt.get('title', '')}': "
            f"{yt.get('summary', yt.get('description', ''))[:400]}",
            doc_type="creator_take",
            source="youtube",
            channel=yt.get("channel", "")
        )

    # Batch upsert into ChromaDB
    if docs:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return len(docs)


# ── RAG Query ─────────────────────────────────────────────────────────────────

def query_pregame_brief(
    matchup: str,
    game_id: Optional[str] = None,
    question: Optional[str] = None,
    n_results: int = 12,
) -> str:
    """
    Main RAG query. Retrieves relevant context from ChromaDB and
    asks the LLM to generate a pre-game briefing.

    Args:
        matchup: e.g. "Red Sox vs Brewers"
        game_id: filter to specific game (optional)
        question: custom question, or None for full pre-game brief
        n_results: number of context chunks to retrieve

    Returns: LLM-generated pre-game analysis as a string
    """
    collection = get_collection()

    # Build query — what we're searching for in the vector store
    query_text = question or (
        f"pre-game analysis starting pitcher lineup injuries umpire "
        f"batter weaknesses matchup prediction for {matchup}"
    )

    # Retrieve relevant chunks
    where_filter = {"game_id": game_id} if game_id else None
    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count() or 1),
        where=where_filter,
    )

    # Format retrieved context
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not chunks:
        return f"No data found for {matchup}. Please run the data ingestion first."

    context_blocks = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        doc_type = meta.get("type", "unknown")
        context_blocks.append(f"[{doc_type.upper()}]\n{chunk}")

    context = "\n\n".join(context_blocks)

    # Build prompt
    system_prompt = """You are Baseball IQ — an elite baseball analyst platform that 
combines Statcast data, injury reports, umpire tendencies, fan sentiment, and 
creator takes to generate sharp pre-game briefings.

Your output should feel like a cross between a seasoned beat reporter and a 
data analyst. Be specific with numbers. Flag real risks. Surface the hidden story.
Structure your brief clearly with sections."""

    user_prompt = f"""Using the following retrieved data, generate a comprehensive 
pre-game briefing for: {matchup}

RETRIEVED CONTEXT:
{context}

Generate a pre-game brief with these sections:
1. **Matchup Overview** — Quick snapshot of the game
2. **Starting Pitchers** — Arsenal, recent form, key stats, what to watch
3. **Lineup Analysis** — Who's hot, who's cold, key matchup advantages
4. **Injury Report** — Who's in/out and how it affects the game
5. **Umpire Card** — Tonight's ump tendencies and what it means for both teams
6. **The Hidden Story** — One underrated angle most fans will miss
7. **Prediction** — Who wins and why, with confidence level

Be specific. Use the numbers from the data. Don't be generic."""

    return _call_llm(system_prompt, user_prompt)


def query_custom(question: str, n_results: int = 8) -> str:
    """
    Answer any baseball question using the RAG knowledge base.
    E.g. "How does Rafael Devers hit against left-handed pitchers?"
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

    system_prompt = "You are Baseball IQ, a sharp baseball analyst. Answer concisely using only the provided data."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    return _call_llm(system_prompt, user_prompt)


# ── LLM Call ──────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def _call_ollama(system: str, user: str) -> str:
    """Call a local Ollama model via its REST API."""
    # Truncate user prompt to stay within a small context window
    max_chars = 3000
    if len(user) > max_chars:
        user = user[:max_chars] + "\n...[truncated]"

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_ctx": 2048,   # smaller KV cache = ~64 MiB vs 128 MiB
                "num_predict": 512, # limit response length
            },
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_llm(system: str, user: str) -> str:
    """
    Call the LLM with the following priority:
      1. Ollama (local, free) — if reachable
      2. Claude (Anthropic API) — if ANTHROPIC_API_KEY is set
      3. Preview stub — prints prompts so you can inspect them
    """
    # 1. Try Ollama
    if REQUESTS_AVAILABLE:
        try:
            return _call_ollama(system, user)
        except Exception as e:
            print(f"   ⚠️  Ollama unavailable ({e}). Trying Anthropic...")

    # 2. Try Anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_AVAILABLE and api_key:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text
        except Exception as e:
            print(f"   ⚠️  Anthropic API error: {e}")

    # 3. Fallback preview
    return (
        "⚠️  No LLM available. Install Ollama (https://ollama.com) and run:\n"
        "   ollama pull llama3\n\n"
        "SYSTEM PROMPT PREVIEW:\n" + system[:200] + "...\n\n"
        "USER PROMPT PREVIEW (first 500 chars):\n" + user[:500] + "..."
    )


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Returns info about what's currently in the vector store."""
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"total_documents": 0, "message": "No data ingested yet."}

    # Sample metadata to show what's in there
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
    print("🧠 Baseball IQ — RAG Pipeline Demo")
    print("=" * 60)

    # Mock game data (replaces real scraped data for testing)
    mock_game_data = {
        "game": {
            "game_id": "999001",
            "date": "2026-04-08",
            "away_team": "Boston Red Sox",
            "home_team": "Milwaukee Brewers",
            "venue": "American Family Field",
            "home_probable_pitcher": {"name": "Freddy Peralta", "id": 669302},
            "away_probable_pitcher": {"name": "Brayan Bello", "id": 678594},
            "status": "Scheduled",
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
                "overall": {
                    "chase_rate": 28.4,
                    "contact_rate": 74.2,
                    "hard_hit_pct": 52.1,
                    "avg_exit_velocity": 93.8,
                },
                "weakness_summary": [
                    "Struggles vs SL — 34.2% whiff rate on 89 pitches seen",
                    "Hits only .178 against sliders away",
                ],
            },
            {
                "batter": "Christian Yelich",
                "overall": {
                    "chase_rate": 24.1,
                    "contact_rate": 77.8,
                    "hard_hit_pct": 44.3,
                    "avg_exit_velocity": 91.2,
                },
                "weakness_summary": [
                    "Hits only .182 against sinkers down in zone",
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
            "accuracy_pct": "93.1",
            "narrative": (
                "Dan Bellino is above-average in accuracy at 93.1%. "
                "He runs a tight zone on the outer half but is consistent. "
                "Good framing catchers benefit in his games. "
                "Pitchers with sharp horizontal movement get favorable calls."
            ),
        },
        "reddit_posts": [
            {
                "subreddit": "r/redsox",
                "title": "Bello needs to get out of the 2nd inning habit",
                "body": "He's given up the most runs in the 2nd inning of any starter this year. Classic slow starter.",
            },
        ],
        "youtube_summaries": [
            {
                "channel": "Jomboy Media",
                "title": "Brewers are sneaky good in April — here's why",
                "summary": "Talkin' Baseball broke down how Milwaukee's bullpen has a 2.14 ERA in April over the last 3 seasons. They're a nightmare late-game opponent right now.",
            },
        ],
    }

    # Ingest
    print("\n📥 Ingesting mock game data...")
    n = ingest_game_data(mock_game_data)
    print(f"   Stored {n} documents in ChromaDB")

    # Stats
    print("\n📊 Vector store contents:")
    stats = get_collection_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Query
    print("\n🔍 Generating pre-game brief...")
    print("-" * 60)
    brief = query_pregame_brief("Boston Red Sox vs Milwaukee Brewers")
    print(brief)