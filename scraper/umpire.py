"""
umpire.py
---------
Scrapes umpire accuracy data from UmpScorecards.com.
Refactored for async execution and Streamlit caching.
"""

import aiohttp
import asyncio
import streamlit as st
from bs4 import BeautifulSoup
import re
from typing import Optional

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _name_to_slug(name: str) -> str:
    """Convert 'Angel Hernandez' → 'Angel_Hernandez'"""
    return name.strip().replace(" ", "_")

def _extract_tendencies(soup: BeautifulSoup) -> list[str]:
    tendencies = []
    for p in soup.find_all(["p", "li", "span"]):
        text = p.get_text(strip=True)
        if any(kw in text.lower() for kw in [
            "tends to", "known for", "frequently", "rarely",
            "high zone", "low zone", "tight", "wide", "inside", "outside"
        ]):
            if 10 < len(text) < 200:
                tendencies.append(text)
    return tendencies[:5]

def _build_narrative(data: dict) -> str:
    name = data.get("umpire", "This umpire")
    accuracy = data.get("accuracy_pct")
    favor = data.get("favor_score")
    impact = data.get("impact_runs")
    tendencies = data.get("tendencies", [])

    lines = []
    if accuracy:
        lines.append(f"{name} has a ball/strike accuracy of {accuracy}% this season.")
    else:
        lines.append(f"{name}'s accuracy data is available at umpscorecards.com.")

    if favor is not None:
        direction = "home team" if float(favor) > 0 else "away team"
        lines.append(f"Their favor score of {favor} suggests a slight lean toward the {direction}.")

    if impact is not None:
        lines.append(f"Missed calls have resulted in approximately {impact} impact runs this season.")

    if tendencies:
        lines.append(tendencies[0])

    return " ".join(lines)

def _parse_umpire_page(html: str, umpire_name: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "umpire": umpire_name,
        "source": "umpscorecards.com",
    }

    stat_blocks = soup.find_all(class_=re.compile(r"stat|card|metric", re.I))

    for block in stat_blocks:
        text = block.get_text(separator=" ", strip=True).lower()
        if "accuracy" in text:
            nums = re.findall(r"\d+\.?\d*%?", text)
            if nums: result["accuracy_pct"] = nums[0].replace("%", "")
        if "correct call" in text:
            nums = re.findall(r"\d+", text)
            if nums: result["correct_calls"] = nums[0]
        if "missed call" in text or "incorrect" in text:
            nums = re.findall(r"\d+", text)
            if nums: result["missed_calls"] = nums[0]
        if "favor" in text:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums: result["favor_score"] = float(nums[0])
        if "impact" in text and "run" in text:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums: result["impact_runs"] = float(nums[0])

    result["tendencies"] = _extract_tendencies(soup)
    result["narrative"] = _build_narrative(result)
    return result

# ── Known Umpire Tendencies (Hardcoded Fallback) ──────────────────────────────

KNOWN_UMPIRE_PROFILES = {
    "Angel Hernandez": {
        "umpire": "Angel Hernandez",
        "accuracy_pct": "91.2",
        "note": "Historically one of the most complained-about umpires; retired after 2023.",
        "tendencies": ["Wide zone on inside corner", "Inconsistent high/low calls"],
        "narrative": "Angel Hernandez historically posted accuracy in the 91% range. He was known for a wide inside corner and inconsistent calls on high pitches.",
    },
    "CB Bucknor": {
        "umpire": "CB Bucknor",
        "accuracy_pct": "91.5",
        "tendencies": ["Slow to call strikeouts", "Tends to expand zone late in counts"],
        "narrative": "CB Bucknor typically runs below-average accuracy (~91.5%) and is known for expanding the zone in hitter's counts (2-0, 3-1).",
    },
}

def get_umpire_fallback(umpire_name: str) -> dict:
    for key, profile in KNOWN_UMPIRE_PROFILES.items():
        if umpire_name.lower() in key.lower() or key.lower() in umpire_name.lower():
            return profile

    return {
        "umpire": umpire_name,
        "accuracy_pct": "92.8",
        "tendencies": [],
        "narrative": f"{umpire_name} is an active MLB umpire. The MLB average ball/strike accuracy is ~92.8%. Scorecards are available at umpscorecards.com.",
    }

# ── Async API Calls ──────────────────────────────────────────────────────────


async def get_umpire_profile(_session: aiohttp.ClientSession, umpire_name: str) -> dict:
    slug = _name_to_slug(umpire_name)
    url = f"https://umpscorecards.com/umpires/?name={slug}"

    try:
        async with _session.get(url, headers=HEADERS, timeout=10) as resp:
            resp.raise_for_status()
            html = await resp.text()
            # Push CPU-bound parsing to a thread to keep async loop fast
            return await asyncio.to_thread(_parse_umpire_page, html, umpire_name)
    except Exception as e:
        return {"error": str(e), "umpire": umpire_name}


async def get_game_umpire_scorecard(_session: aiohttp.ClientSession, game_pk: int) -> dict:
    url = f"https://umpscorecards.com/games/?id={game_pk}"
    try:
        async with _session.get(url, headers=HEADERS, timeout=10) as resp:
            resp.raise_for_status()
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        return {"error": str(e), "game_pk": game_pk}

    return {
        "game_pk": game_pk,
        "source": url,
        "raw_text_preview": soup.get_text(separator="\n", strip=True)[:500],
    }


async def get_umpire_info(_session: aiohttp.ClientSession, umpire_name: str, try_scrape: bool = True) -> dict:
    """Main entry point. Tries live scrape first, falls back to hardcoded data."""
    if try_scrape:
        try:
            result = await get_umpire_profile(_session, umpire_name)
            if "error" not in result:
                return result
        except Exception:
            pass
    return get_umpire_fallback(umpire_name)