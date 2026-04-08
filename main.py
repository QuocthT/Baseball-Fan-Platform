"""
main.py
-------
Orchestrator: scrape everything for a given game, ingest into RAG, generate brief.

Usage:
    python main.py                          # today's games
    python main.py --date 2026-04-08        # specific date
    python main.py --game "Red Sox Brewers" # filter to specific matchup
    python main.py --query "How does Devers hit against sliders?"
"""

import argparse
import json
from datetime import date

from scraper.mlb_api import (
    get_games, get_lineup, get_pitcher_stats,
    get_injuries, get_team_id, get_bullpen
)
from scraper.statcast import (
    get_pitcher_arsenal, get_batter_profile, get_recent_batter_form
)
from scraper.umpire import get_umpire_info
from pipeline.rag import ingest_game_data, query_pregame_brief, query_custom, get_collection_stats


def build_game_data(game: dict) -> dict:
    """Assemble full data bundle for one game."""
    print(f"\n  📡 Fetching data for: {game['away_team']} @ {game['home_team']}")

    data = {"game": game}

    # Pitcher stats
    for side in ["home", "away"]:
        pitcher = game[f"{side}_probable_pitcher"]
        if pitcher["id"]:
            print(f"     ⚾ {side} pitcher: {pitcher['name']}")
            data[f"{side}_pitcher_stats"] = get_pitcher_stats(pitcher["id"])
            data[f"{side}_pitcher_arsenal"] = get_pitcher_arsenal(pitcher["name"])

    # Lineup (available ~1hr before game)
    lineup = get_lineup(game["game_id"])
    data["home_lineup"] = lineup.get("home", {})
    data["away_lineup"] = lineup.get("away", {})

    # Batter profiles for top of lineup
    batter_profiles = []
    for side in ["home", "away"]:
        batters = data.get(f"{side}_lineup", {}).get("batting_order", [])
        for batter in batters[:4]:  # top 4 in order
            print(f"     👤 Profiling: {batter['name']}")
            profile = get_batter_profile(batter["name"])
            if "error" not in profile:
                batter_profiles.append(profile)
    data["batter_profiles"] = batter_profiles

    # Injuries for both teams
    injuries = []
    for side in ["home", "away"]:
        team_name = game[f"{side}_team"]
        team_id = get_team_id(team_name)
        if team_id:
            injuries.extend(get_injuries(team_id=team_id, days_back=14))
    data["injuries"] = injuries

    # Umpire (you'd normally get this from the game feed)
    # MLB API includes umpire in /game/{id}/linescore for live games
    data["umpire"] = get_umpire_info("TBD", try_scrape=False)

    # Reddit / YouTube — import here to avoid circular
    try:
        from scraper.reddit_youtube import get_reddit_posts, get_youtube_summaries
        matchup_query = f"{game['away_team']} {game['home_team']}"
        data["reddit_posts"] = get_reddit_posts(matchup_query, limit=5)
        data["youtube_summaries"] = get_youtube_summaries(matchup_query, max_results=3)
    except Exception as e:
        print(f"     ⚠️  Reddit/YouTube skipped: {e}")
        data["reddit_posts"] = []
        data["youtube_summaries"] = []

    return data


def run(game_date: str = None, game_filter: str = None, query: str = None):
    if query:
        print(f"\n🔍 Custom query: {query}")
        print("-" * 60)
        print(query_custom(query))
        return

    game_date = game_date or date.today().isoformat()
    print(f"\n🗓️  Baseball IQ — {game_date}")
    print("=" * 60)

    games = get_games(game_date)
    if not games:
        print("No games found.")
        return

    print(f"Found {len(games)} games")
    if game_filter:
        games = [g for g in games if
                 game_filter.lower() in g["home_team"].lower() or
                 game_filter.lower() in g["away_team"].lower()]
        print(f"Filtered to {len(games)} matching games")

    for game in games:
        game_data = build_game_data(game)
        n_docs = ingest_game_data(game_data)
        matchup = f"{game['away_team']} @ {game['home_team']}"
        print(f"\n  ✅ Ingested {n_docs} documents for {matchup}")

        print(f"\n{'='*60}")
        print(f"🧠 PRE-GAME BRIEF: {matchup}")
        print(f"{'='*60}")
        brief = query_pregame_brief(matchup, game_id=str(game["game_id"]))
        print(brief)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseball IQ Pre-Game Briefing")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--game", help="Filter to games matching this team name")
    parser.add_argument("--query", help="Ask a custom question against the knowledge base")
    args = parser.parse_args()

    run(game_date=args.date, game_filter=args.game, query=args.query)