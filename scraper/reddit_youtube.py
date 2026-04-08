"""
reddit_youtube.py
-----------------
Fan sentiment from Reddit and baseball creator content from YouTube.

Reddit:  Uses PRAW (Python Reddit API Wrapper) — free dev account at reddit.com/prefs/apps
YouTube: Uses YouTube Data API v3 — free tier at console.cloud.google.com (10k units/day)

Both are optional — the RAG pipeline degrades gracefully without them.
"""

import os
import re
from typing import Optional
from datetime import datetime, timedelta


# ── Reddit ────────────────────────────────────────────────────────────────────

BASEBALL_SUBREDDITS = [
    "baseball", "redsox", "brewers", "mlb",
    # Add team-specific ones as needed
]

# Creators whose channels we trust for baseball analysis
TRUSTED_CHANNELS = {
    "UCylmMBFmjuKoEJPEDZJDgLQ": "Jomboy Media",
    "UCdHTCaFfHbFLiGSA_KsMqpg": "Foolish Baseball",
    "UCdRSRAFKyHVZh7XsB6ij7lg": "Pitching Ninja",   # Rob Friedman
    "UCO3-hFmSCZQBtj7rCLJXWnA": "Just Baseball",
    "UCzg1q5gQl5tkZYJI2u5QKaQ": "Baseball Doesn't Exist",
}


def get_reddit_posts(query: str, limit: int = 10,
                     subreddits: Optional[list] = None) -> list[dict]:
    """
    Search Reddit for posts related to a matchup or player.

    Args:
        query: e.g. "Red Sox Brewers" or "Rafael Devers injury"
        limit: max posts to return
        subreddits: list of subreddit names (defaults to BASEBALL_SUBREDDITS)

    Setup:
        1. Go to reddit.com/prefs/apps
        2. Create a "script" app
        3. Add REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT to .env
    """
    try:
        import praw
    except ImportError:
        return [{"error": "praw not installed. Run: pip install praw"}]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "BaseballIQ/1.0")

    if not client_id or not client_secret:
        return _mock_reddit_posts(query)

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subs = subreddits or BASEBALL_SUBREDDITS
    combined = "+".join(subs)
    subreddit = reddit.subreddit(combined)

    posts = []
    cutoff = datetime.utcnow() - timedelta(days=3)  # last 3 days only

    for submission in subreddit.search(query, sort="new", limit=limit * 3):
        created = datetime.utcfromtimestamp(submission.created_utc)
        if created < cutoff:
            continue

        # Skip low-engagement posts
        if submission.score < 5 and submission.num_comments < 3:
            continue

        posts.append({
            "id": submission.id,
            "subreddit": f"r/{submission.subreddit.display_name}",
            "title": submission.title,
            "body": submission.selftext[:500] if submission.selftext else "",
            "score": submission.score,
            "comments": submission.num_comments,
            "url": f"https://reddit.com{submission.permalink}",
            "created": created.isoformat(),
            "flair": submission.link_flair_text or "",
        })

        if len(posts) >= limit:
            break

    return posts


def _mock_reddit_posts(query: str) -> list[dict]:
    """Mock data for when Reddit credentials aren't configured."""
    return [
        {
            "subreddit": "r/baseball",
            "title": f"[Game Thread] {query}",
            "body": "Mock Reddit post. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env for real data.",
            "score": 42,
            "comments": 18,
            "source": "mock",
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
    print("📱 REDDIT POSTS (mock — add credentials to .env for real data)")
    posts = get_reddit_posts("Red Sox Brewers")
    for p in posts:
        print(f"  [{p['subreddit']}] {p['title']}")

    print("\n📺 YOUTUBE SUMMARIES (mock — add YOUTUBE_API_KEY for real data)")
    videos = get_youtube_summaries("Red Sox Brewers preview")
    for v in videos:
        print(f"  [{v['channel']}] {v['title']}")
        print(f"    → {v['summary']}")