"""
umpire.py
---------
Scrapes umpire accuracy data from UmpScorecards.com — the best public
source for per-umpire, per-game ball/strike accuracy statistics.

Data includes:
- Overall accuracy %
- Correct calls / missed calls
- Favor score (positive = favors home team)
- Strike zone bias (tight / wide / high / low)
- Impact runs (how many runs the missed calls cost)
"""

import requests
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


# ── Main Umpire Lookup ────────────────────────────────────────────────────────

def get_umpire_profile(umpire_name: str) -> dict:
    """
    Returns a season-level umpire profile from UmpScorecards.

    Args:
        umpire_name: Full name, e.g. "Angel Hernandez" or "CB Bucknor"

    Returns dict with accuracy, bias tendencies, and narrative summary.
    """
    # UmpScorecards uses slugified names in URLs
    slug = _name_to_slug(umpire_name)
    url = f"https://umpscorecards.com/umpires/?name={slug}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e), "umpire": umpire_name}

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_umpire_page(soup, umpire_name)


def _name_to_slug(name: str) -> str:
    """Convert 'Angel Hernandez' → 'Angel_Hernandez'"""
    return name.strip().replace(" ", "_")


def _parse_umpire_page(soup: BeautifulSoup, umpire_name: str) -> dict:
    """
    Parse the UmpScorecards umpire page.
    The site uses React-rendered content, so we target the data attributes
    and any server-side rendered stat blocks.
    """
    result = {
        "umpire": umpire_name,
        "source": "umpscorecards.com",
    }

    # Try to extract stat cards — the site renders key stats in labeled divs
    stat_blocks = soup.find_all(class_=re.compile(r"stat|card|metric", re.I))

    for block in stat_blocks:
        text = block.get_text(separator=" ", strip=True).lower()

        if "accuracy" in text:
            nums = re.findall(r"\d+\.?\d*%?", text)
            if nums:
                result["accuracy_pct"] = nums[0].replace("%", "")

        if "correct call" in text:
            nums = re.findall(r"\d+", text)
            if nums:
                result["correct_calls"] = nums[0]

        if "missed call" in text or "incorrect" in text:
            nums = re.findall(r"\d+", text)
            if nums:
                result["missed_calls"] = nums[0]

        if "favor" in text:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums:
                result["favor_score"] = float(nums[0])

        if "impact" in text and "run" in text:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums:
                result["impact_runs"] = float(nums[0])

    # Narrative tendencies — look for descriptive text blocks
    result["tendencies"] = _extract_tendencies(soup)
    result["narrative"] = _build_narrative(result)

    return result


def _extract_tendencies(soup: BeautifulSoup) -> list[str]:
    """Pull any qualitative tendency bullets from the page."""
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
    """
    Builds a 2-3 sentence human-readable umpire scouting note.
    This is what gets fed into the RAG pipeline as context.
    """
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


# ── Game-Level Umpire Data ────────────────────────────────────────────────────

def get_game_umpire_scorecard(game_pk: int) -> dict:
    """
    Fetches the scorecard for a specific game by MLB game ID.
    UmpScorecards has individual game pages.
    """
    url = f"https://umpscorecards.com/games/?id={game_pk}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return {"error": str(e), "game_pk": game_pk}

    return {
        "game_pk": game_pk,
        "source": url,
        "raw_text_preview": soup.get_text(separator="\n", strip=True)[:500],
    }


# ── Known Umpire Tendencies (Hardcoded Fallback) ──────────────────────────────
# Useful when scraping fails or for offline demo / class presentation.
# Based on publicly available Statcast + UmpScorecards research.

KNOWN_UMPIRE_PROFILES = {
    "Angel Hernandez": {
        "umpire": "Angel Hernandez",
        "accuracy_pct": "91.2",
        "note": "Historically one of the most complained-about umpires; "
                "retired after 2023 discrimination lawsuit. No longer active.",
        "tendencies": ["Wide zone on inside corner", "Inconsistent high/low calls"],
        "narrative": (
            "Angel Hernandez historically posted accuracy in the 91% range, "
            "below the MLB average of ~93%. He was known for a wide inside corner "
            "and inconsistent calls on high pitches."
        ),
    },
    "CB Bucknor": {
        "umpire": "CB Bucknor",
        "accuracy_pct": "91.5",
        "tendencies": ["Slow to call strikeouts", "Tends to expand zone late in counts"],
        "narrative": (
            "CB Bucknor typically runs below-average accuracy (~91.5%) "
            "and is known for expanding the zone in hitter's counts (2-0, 3-1). "
            "Pitchers benefit from working ahead against his games."
        ),
    },
    "Ángel Hernández": {
        "umpire": "Ángel Hernández",
        "accuracy_pct": "91.2",
        "tendencies": ["Wide zone on inside corner"],
        "narrative": "See Angel Hernandez.",
    },
}


def get_umpire_fallback(umpire_name: str) -> dict:
    """
    Returns hardcoded profile if name matches known umpires.
    Falls back to a generic average-umpire profile.
    """
    for key, profile in KNOWN_UMPIRE_PROFILES.items():
        if umpire_name.lower() in key.lower() or key.lower() in umpire_name.lower():
            return profile

    # Generic fallback
    return {
        "umpire": umpire_name,
        "accuracy_pct": "92.8",  # 2025 MLB average per FanGraphs
        "tendencies": [],
        "narrative": (
            f"{umpire_name} is an active MLB umpire. "
            f"The 2025 MLB average ball/strike accuracy is 92.8% per Statcast. "
            f"Detailed per-game scorecards are available at umpscorecards.com. "
            f"In 2026, the ABS challenge system allows each team 2 challenges per game."
        ),
    }


def get_umpire_info(umpire_name: str, try_scrape: bool = True) -> dict:
    """
    Main entry point. Tries live scrape first, falls back to hardcoded data.
    """
    if try_scrape:
        try:
            result = get_umpire_profile(umpire_name)
            if "error" not in result:
                return result
        except Exception:
            pass

    return get_umpire_fallback(umpire_name)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧑‍⚖️ UMPIRE PROFILES")
    print("=" * 60)

    test_umpires = ["CB Bucknor", "Joe West", "Dan Bellino"]
    for name in test_umpires:
        profile = get_umpire_info(name, try_scrape=False)
        print(f"\n{profile['umpire']}")
        print(f"  Accuracy: {profile.get('accuracy_pct', 'N/A')}%")
        print(f"  {profile.get('narrative', '')}")