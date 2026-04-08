"""
reddit_youtube.py
-----------------
Fan sentiment from Reddit and baseball creator content from YouTube.

Reddit:  Scraped directly via Reddit's public JSON endpoints — no API key needed.
         Reddit exposes <url>.json for any listing or search page. We set a
         descriptive User-Agent to comply with Reddit's API rules.

YouTube: Uses YouTube Data API v3 — free tier at console.cloud.google.com (10k units/day)
         Falls back to mock data when YOUTUBE_API_KEY is not set.
"""

import os
import re
import requests
from typing import Optional
from datetime import datetime, timedelta


# ── Reddit ────────────────────────────────────────────────────────────────────

# Subreddits to search across (no credentials needed)
BASEBALL_SUBREDDITS = [
    "baseball", "mlb",
    # Team-specific subs are added dynamically based on matchup teams
]

# Per-team subreddit mapping (lowercase last word of team name → subreddit)
TEAM_SUBREDDITS = {
    "yankees":     "NYYankees",
    "redsox":      "redsox",
    "mets":        "NewYorkMets",
    "dodgers":     "Dodgers",
    "giants":      "SFGiants",
    "cubs":        "chicagocubs",
    "cardinals":   "Cardinals",
    "brewers":     "Brewers",
    "reds":        "reds",
    "pirates":     "buccos",
    "phillies":    "phillies",
    "braves":      "Braves",
    "marlins":     "letsgofish",
    "nationals":   "Nationals",
    "astros":      "Astros",
    "rangers":     "TexasRangers",
    "angels":      "angelsbaseball",
    "athletics":   "OaklandAthletics",
    "mariners":    "Mariners",
    "padres":      "Padres",
    "rockies":     "ColoradoRockies",
    "diamondbacks":"azdiamondbacks",
    "tigers":      "motorcitykitties",
    "indians":     "WahoosTipi",
    "guardians":   "ClevelandGuardians",
    "whitesox":    "whitesox",
    "twins":       "minnesotatwins",
    "royals":      "KCRoyals",
    "orioles":     "orioles",
    "rays":        "TampaBayRays",
    "bluejays":    "Torontobluejays",
    "jays":        "Torontobluejays",
}

_REDDIT_HEADERS = {
    "User-Agent": "BaseballIQ/1.0 (baseball fan dashboard; educational project)",
    "Accept": "application/json",
}

_REDDIT_BASE = "https://www.reddit.com"


def _reddit_get(url: str, params: dict) -> list[dict]:
    """
    Fetch a Reddit JSON listing and return the list of post data dicts.
    Returns [] on any error so callers degrade gracefully.
    """
    try:
        resp = requests.get(url, params=params, headers=_REDDIT_HEADERS, timeout=10)
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
        return [c["data"] for c in children]
    except Exception:
        return []


def get_reddit_posts(query: str, limit: int = 10,
                     subreddits: Optional[list] = None) -> list[dict]:
    """
    Search Reddit for posts related to a matchup or player.
    Uses Reddit's public JSON API — no credentials required.

    Args:
        query:      e.g. "Red Sox Brewers" or "Rafael Devers injury"
        limit:      max posts to return
        subreddits: override the default list of subreddits to search
    """
    subs = list(subreddits) if subreddits else list(BASEBALL_SUBREDDITS)

    # Auto-add team-specific subs based on words in the query
    for word in query.lower().split():
        if word in TEAM_SUBREDDITS and TEAM_SUBREDDITS[word] not in subs:
            subs.append(TEAM_SUBREDDITS[word])

    cutoff = datetime.utcnow() - timedelta(days=4)
    seen_ids: set[str] = set()
    posts: list[dict] = []

    # Search the combined multireddit (e.g. r/baseball+mlb+Brewers)
    combined = "+".join(subs)
    raw = _reddit_get(
        f"{_REDDIT_BASE}/r/{combined}/search.json",
        params={"q": query, "sort": "new", "restrict_sr": "1",
                "limit": min(limit * 4, 100), "t": "week"},
    )

    # Fall back to sitewide search if the multireddit search returned nothing
    if not raw:
        raw = _reddit_get(
            f"{_REDDIT_BASE}/search.json",
            params={"q": query, "sort": "new", "limit": min(limit * 4, 100), "t": "week"},
        )

    for item in raw:
        post_id = item.get("id", "")
        if post_id in seen_ids:
            continue
        seen_ids.add(post_id)

        created_utc = item.get("created_utc", 0)
        created = datetime.utcfromtimestamp(created_utc)
        if created < cutoff:
            continue

        # Skip spam / zero-engagement posts
        if item.get("score", 0) < 1 and item.get("num_comments", 0) < 1:
            continue

        body = item.get("selftext", "") or ""
        # "[removed]" / "[deleted]" bodies are useless
        if body.lower() in ("[removed]", "[deleted]"):
            body = ""

        posts.append({
            "id":       post_id,
            "sub":      f"r/{item.get('subreddit', 'baseball')}",
            "title":    item.get("title", ""),
            "body":     body[:500],
            "score":    item.get("score", 0),
            "comments": item.get("num_comments", 0),
            "url":      f"{_REDDIT_BASE}{item.get('permalink', '')}",
            "created":  created.isoformat(),
            "flair":    item.get("link_flair_text") or "",
        })

        if len(posts) >= limit:
            break

    if not posts:
        return _fallback_reddit_posts(query)

    return posts


def _fallback_reddit_posts(query: str) -> list[dict]:
    """Shown when scraping returns nothing (e.g. rate-limited or no matching posts)."""
    return [
        {
            "sub":      "r/baseball",
            "title":    f"[Game Thread] {query}",
            "body":     "No recent Reddit posts found. Reddit may be rate-limiting — try again shortly.",
            "score":    0,
            "comments": 0,
        }
    ]


# ── YouTube ───────────────────────────────────────────────────────────────────

def get_youtube_summaries(query: str, max_results: int = 5,
                           days_back: int = 7) -> list[dict]:
    """
    Searches YouTube for recent baseball analysis videos from trusted channels.
    Returns video metadata + description (transcripts require extra setup).

    Args:
        query: e.g. "Red Sox Brewers preview" or "Freddy Peralta"
        max_results: max videos to return
        days_back: only return videos published in last N days

    Setup:
        1. Go to console.cloud.google.com
        2. Enable "YouTube Data API v3"
        3. Create an API key
        4. Add YOUTUBE_API_KEY to .env
        Free tier: 10,000 units/day (each search = 100 units → 100 searches/day)
    """
    import requests

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return _mock_youtube_results(query)

    published_after = (
        datetime.utcnow() - timedelta(days=days_back)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Search across all trusted channels
    all_videos = []

    for channel_id, channel_name in TRUSTED_CHANNELS.items():
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "key": api_key,
                    "channelId": channel_id,
                    "q": query,
                    "part": "snippet",
                    "type": "video",
                    "order": "date",
                    "publishedAfter": published_after,
                    "maxResults": 2,
                },
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])

            for item in items:
                snippet = item.get("snippet", {})
                all_videos.append({
                    "video_id": item["id"]["videoId"],
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", "")[:400],
                    "published": snippet.get("publishedAt", ""),
                    "url": f"https://youtube.com/watch?v={item['id']['videoId']}",
                    "summary": _extract_key_points(snippet.get("description", "")),
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url"),
                })

        except Exception as e:
            print(f"    YouTube error for {channel_name}: {e}")
            continue

    # Sort by published date, return top results
    all_videos.sort(key=lambda x: x.get("published", ""), reverse=True)
    return all_videos[:max_results]


def _extract_key_points(description: str) -> str:
    """
    Extract key sentences from a video description.
    In a full implementation, you'd pull the transcript using youtube-transcript-api.
    """
    if not description:
        return ""
    # Return first 2 sentences of description as a rough summary
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    return " ".join(sentences[:2])


def _mock_youtube_results(query: str) -> list[dict]:
    """Mock data for when YouTube API key isn't configured."""
    return [
        {
            "channel": "Jomboy Media",
            "title": f"Breaking down tonight's {query} matchup",
            "description": "Mock YouTube result. Add YOUTUBE_API_KEY to .env for real data.",
            "summary": "Add YOUTUBE_API_KEY to .env to pull real creator content.",
            "url": "https://youtube.com",
            "source": "mock",
        },
        {
            "channel": "Foolish Baseball",
            "title": f"Should you be worried about {query}?",
            "description": "Deep dive into the numbers.",
            "summary": "Configure YouTube API to surface real analyst takes.",
            "url": "https://youtube.com",
            "source": "mock",
        }
    ]


# ── YouTube Transcripts (bonus) ───────────────────────────────────────────────

def get_video_transcript(video_id: str) -> str:
    """
    Fetch the auto-generated transcript for a YouTube video.
    This is gold for RAG — you get the actual spoken analysis.

    Requires: pip install youtube-transcript-api
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript_list])
        return full_text[:3000]  # cap at 3000 chars for RAG
    except ImportError:
        return "Install youtube-transcript-api for transcript support."
    except Exception as e:
        return f"Transcript unavailable: {e}"


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📱 REDDIT POSTS (scraped — no credentials needed)")
    posts = get_reddit_posts("Red Sox Brewers", limit=5)
    for p in posts:
        print(f"  [{p['sub']}] {p['title']}")
        print(f"    ▲ {p['score']}  💬 {p['comments']}  — {p['url']}")

    print("\n📺 YOUTUBE SUMMARIES (mock — add YOUTUBE_API_KEY for real data)")
    videos = get_youtube_summaries("Red Sox Brewers preview")
    for v in videos:
        print(f"  [{v['channel']}] {v['title']}")
        print(f"    → {v['summary']}")