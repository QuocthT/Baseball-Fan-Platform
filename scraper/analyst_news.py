"""
analyst_news.py
---------------
Scrapes recent MLB news and written analyst reports from RSS feeds.
Currently configured to pull team-specific feeds from MLB Trade Rumors.
"""

import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re

def _slugify(team_name: str) -> str:
    """Converts 'Boston Red Sox' to 'boston-red-sox' for URLs."""
    # Handle team names with special characters or drop the city if needed,
    # but MLBTR usually uses the full hyphenated name.
    return team_name.lower().strip().replace(" ", "-")

async def get_team_news(session: aiohttp.ClientSession, team_name: str, limit: int = 3) -> list[dict]:
    """
    Fetches the latest written news and analysis for a specific team.
    """
    slug = _slugify(team_name)
    url = f"https://www.mlbtraderumors.com/{slug}/feed"
    
    try:
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            xml_data = await resp.text()
            
            # We use html.parser as it's built-in and handles basic RSS XML tags well
            soup = BeautifulSoup(xml_data, "html.parser")
            items = soup.find_all("item")
            
            news = []
            for item in items[:limit]:
                title = item.title.text if item.title else "No Title"
                
                # RSS descriptions often have HTML tags (like <a> or <p>) inside them. 
                # We pass it through BeautifulSoup again to strip it to pure text for the LLM.
                raw_desc = item.description.text if item.description else ""
                clean_desc = BeautifulSoup(raw_desc, "html.parser").get_text(separator=" ", strip=True)
                
                # Cut the description down to 400 chars so we don't blow up our LLM context window
                clean_desc = clean_desc[:400] + "..." if len(clean_desc) > 400 else clean_desc
                
                pub_date = item.pubdate.text if item.pubdate else "Recent"
                
                news.append({
                    "team": team_name,
                    "title": title,
                    "summary": clean_desc,
                    "published": pub_date
                })
                
            return news
            
    except Exception as e:
        print(f"⚠️ Could not fetch news for {team_name}: {e}")
        return [{
            "team": team_name, 
            "title": f"Live news unavailable for {team_name}", 
            "summary": "Analyst reports could not be scraped at this time.", 
            "published": "N/A"
        }]