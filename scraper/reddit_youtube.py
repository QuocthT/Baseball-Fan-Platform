"""
reddit_youtube.py
-----------------
Fan sentiment from Reddit and baseball creator content from YouTube.
Refactored for async execution and Streamlit caching.
"""

import os
import re
import aiohttp
import asyncio
import streamlit as st
from typing import Optional
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

BASEBALL_SUBREDDITS = ["baseball", "mlb"]

TEAM_SUBREDDITS = {
    "yankees": "NYYankees", "redsox": "redsox", "mets": "NewYorkMets",
    "dodgers": "Dodgers", "giants": "SFGiants", "cubs": "chicagocubs",
    "brewers": "Brewers", "braves": "Braves", "astros": "Astros",
    "padres": "Padres", "phillies": "phillies", "mariners": "Mariners",
    "orioles": "orioles", "rays": "TampaBayRays", "bluejays": "Torontobluejays",
}

TRUSTED_CHANNELS = {
    "UCO2xKqg23oVf8zG9g6O8p6g": "Jomboy Media",
    "UCaPgR3JkG2mEqO9k4E7kE1A": "Foolish Baseball",
    "UCqL1B4Q0P0O0aM3Ua014_Kw": "Fuzzy"
}

_REDDIT_HEADERS = {
    "User-Agent": "BaseballIQ/1.0 (baseball fan dashboard; educational project)",
    "Accept": "application/json",
}
_REDDIT_BASE = "https://www.reddit.com"

# ── Reddit (Async) ────────────────────────────────────────────────────────────

async def _reddit_get(session: aiohttp.ClientSession, url: str, params: dict) -> list[dict]:
    try:
        async with session.get(url, params=params, headers=_REDDIT_HEADERS, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            children = data["data"]["children"]
            return [c["data"] for c in children]
    except Exception:
        return []

def _fallback_reddit_posts(query: str) -> list[dict]:
    return [{
        "sub": "r/baseball",
        "title": f"[Game Thread] {query}",
        "body": "No recent Reddit posts found. Reddit may be rate-limiting — try again shortly.",
        "score": 0, "comments": 0,
    }]

async def get_reddit_posts(_session: aiohttp.ClientSession, query: str, limit: int = 10, subreddits: Optional[list] = None) -> list[dict]:
    subs = list(subreddits) if subreddits else list(BASEBALL_SUBREDDITS)

    for word in query.lower().split():
        if word in TEAM_SUBREDDITS and TEAM_SUBREDDITS[word] not in subs:
            subs.append(TEAM_SUBREDDITS[word])

    cutoff = datetime.utcnow() - timedelta(days=4)
    seen_ids: set[str] = set()
    posts: list[dict] = []

    combined = "+".join(subs)
    raw = await _reddit_get(_session, f"{_REDDIT_BASE}/r/{combined}/search.json",
        params={"q": query, "sort": "new", "restrict_sr": "1", "limit": min(limit * 4, 100), "t": "week"})

    if not raw:
        raw = await _reddit_get(_session, f"{_REDDIT_BASE}/search.json",
            params={"q": query, "sort": "new", "limit": min(limit * 4, 100), "t": "week"})

    for item in raw:
        post_id = item.get("id", "")
        if post_id in seen_ids: continue
        seen_ids.add(post_id)

        created_utc = item.get("created_utc", 0)
        created = datetime.utcfromtimestamp(created_utc)
        if created < cutoff: continue

        if item.get("score", 0) < 1 and item.get("num_comments", 0) < 1: continue

        body = item.get("selftext", "") or ""
        if body.lower() in ("[removed]", "[deleted]"): body = ""

        posts.append({
            "id": post_id, "sub": f"r/{item.get('subreddit', 'baseball')}",
            "title": item.get("title", ""), "body": body[:500],
            "score": item.get("score", 0), "comments": item.get("num_comments", 0),
            "url": f"{_REDDIT_BASE}{item.get('permalink', '')}",
            "created": created.isoformat(), "flair": item.get("link_flair_text") or "",
        })
        if len(posts) >= limit: break

    return posts if posts else _fallback_reddit_posts(query)

# ── YouTube (Async) ───────────────────────────────────────────────────────────

def _extract_key_points(description: str) -> str:
    if not description: return ""
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    return " ".join(sentences[:2])

def _mock_youtube_results(query: str) -> list[dict]:
    return [
        {"channel": "Jomboy Media", "title": f"Breaking down tonight's {query} matchup", "description": "Mock data.", "summary": "Add YOUTUBE_API_KEY to .env.", "url": "https://youtube.com", "source": "mock"},
        {"channel": "Foolish Baseball", "title": f"Should you be worried about {query}?", "description": "Mock data.", "summary": "Configure YouTube API to surface real analyst takes.", "url": "https://youtube.com", "source": "mock"}
    ]

async def get_youtube_summaries(_session: aiohttp.ClientSession, query: str, max_results: int = 5, days_back: int = 7) -> list[dict]:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return _mock_youtube_results(query)

    published_after = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_videos = []

    # Fetch concurrently for all channels
    async def fetch_channel(channel_id, channel_name):
        try:
            async with _session.get("https://www.googleapis.com/youtube/v3/search", params={
                "key": api_key, "channelId": channel_id, "q": query, "part": "snippet",
                "type": "video", "order": "date", "publishedAfter": published_after, "maxResults": 2,
            }, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return channel_name, channel_id, data.get("items", [])
        except Exception as e:
            print(f"    YouTube error for {channel_name}: {e}")
            return channel_name, channel_id, []

    tasks = [fetch_channel(c_id, c_name) for c_id, c_name in TRUSTED_CHANNELS.items()]
    results = await asyncio.gather(*tasks)

    for c_name, c_id, items in results:
        for item in items:
            snippet = item.get("snippet", {})
            all_videos.append({
                "video_id": item["id"]["videoId"], "channel": c_name, "channel_id": c_id,
                "title": snippet.get("title", ""), "description": snippet.get("description", "")[:400],
                "published": snippet.get("publishedAt", ""), "url": f"https://youtube.com/watch?v={item['id']['videoId']}",
                "summary": _extract_key_points(snippet.get("description", "")),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url"),
            })

    all_videos.sort(key=lambda x: x.get("published", ""), reverse=True)
    return all_videos[:max_results]