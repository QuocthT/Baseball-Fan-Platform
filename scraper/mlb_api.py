"""
mlb_api.py
----------
Pulls live game data from the free, keyless MLB Stats API.
Covers: today's schedule, lineups, probable pitchers, injury transactions.

MLB Stats API base: https://statsapi.mlb.com/api/v1/
No authentication required.
"""

import requests
from datetime import date, timedelta
from typing import Optional
import json

BASE = "https://statsapi.mlb.com/api/v1"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = {}) -> dict:
    url = f"{BASE}{endpoint}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _team_name(team_data: dict) -> str:
    return team_data.get("team", {}).get("name", "Unknown")


# ── Schedule ─────────────────────────────────────────────────────────────────

def get_games(game_date: Optional[str] = None) -> list[dict]:
    """
    Returns all MLB games for a given date (YYYY-MM-DD).
    Defaults to today.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    data = _get("/schedule", params={
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher,team,linescore"
    })

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]

            game = {
                "game_id": g["gamePk"],
                "date": game_date,
                "status": g["status"]["detailedState"],
                "venue": g.get("venue", {}).get("name", "Unknown"),
                "time": g.get("gameDate", ""),
                "home_team": _team_name(home),
                "away_team": _team_name(away),
                "home_probable_pitcher": _extract_pitcher(home),
                "away_probable_pitcher": _extract_pitcher(away),
            }
            games.append(game)

    return games


def _extract_pitcher(team_data: dict) -> dict:
    p = team_data.get("probablePitcher", {})
    if not p:
        return {"name": "TBD", "id": None}
    return {
        "name": p.get("fullName", "TBD"),
        "id": p.get("id"),
    }


# ── Lineups ───────────────────────────────────────────────────────────────────

def get_lineup(game_id: int) -> dict:
    """
    Returns confirmed lineups for both teams in a game.
    Note: lineups are only available ~1 hour before first pitch.
    """
    data = _get(f"/game/{game_id}/boxscore")

    lineups = {}
    for side in ["home", "away"]:
        team_data = data.get("teams", {}).get(side, {})
        team_name = team_data.get("team", {}).get("name", side)
        batters = []

        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        for pid in batting_order:
            key = f"ID{pid}"
            p = players.get(key, {})
            person = p.get("person", {})
            stats = p.get("seasonStats", {}).get("batting", {})
            batters.append({
                "name": person.get("fullName", "Unknown"),
                "id": person.get("id"),
                "position": p.get("position", {}).get("abbreviation", "?"),
                "batting_avg": stats.get("avg", ".???"),
                "ops": stats.get("ops", ".???"),
                "home_runs": stats.get("homeRuns", 0),
            })

        lineups[side] = {
            "team": team_name,
            "batting_order": batters,
        }

    return lineups


# ── Pitcher Stats ─────────────────────────────────────────────────────────────

def get_pitcher_stats(player_id: int, season: int = 2025) -> dict:
    """
    Returns season pitching stats for a given player ID.
    """
    data = _get(f"/people/{player_id}", params={
        "hydrate": f"stats(group=pitching,type=season,season={season})"
    })

    person = data.get("people", [{}])[0]
    name = person.get("fullName", "Unknown")
    stats_list = person.get("stats", [])

    if not stats_list:
        return {"name": name, "stats": {}}

    stats = stats_list[0].get("splits", [{}])[0].get("stat", {})
    return {
        "name": name,
        "id": player_id,
        "season": season,
        "era": stats.get("era", "N/A"),
        "whip": stats.get("whip", "N/A"),
        "strikeouts": stats.get("strikeOuts", "N/A"),
        "innings_pitched": stats.get("inningsPitched", "N/A"),
        "wins": stats.get("wins", "N/A"),
        "losses": stats.get("losses", "N/A"),
        "walks": stats.get("baseOnBalls", "N/A"),
        "hits_allowed": stats.get("hits", "N/A"),
        "strikeout_per_9": stats.get("strikeoutsPer9Inn", "N/A"),
    }


# ── Injury / Transaction Reports ──────────────────────────────────────────────

def get_injuries(team_id: Optional[int] = None, days_back: int = 7) -> list[dict]:
    """
    Pulls recent IL transactions (injuries).
    Optionally filter by team_id. Covers the last `days_back` days.
    """
    start = (date.today() - timedelta(days=days_back)).isoformat()
    end = date.today().isoformat()

    params = {
        "startDate": start,
        "endDate": end,
        "sportId": 1,
        "limit": 200,
    }
    if team_id:
        params["teamId"] = team_id

    data = _get("/transactions", params=params)
    transactions = data.get("transactions", [])

    injuries = []
    for t in transactions:
        type_code = t.get("typeCode", "")
        # IL = injured list placements/activations
        if "IL" in type_code or "INJURY" in type_code.upper() or "DL" in type_code:
            injuries.append({
                "player": t.get("player", {}).get("fullName", "Unknown"),
                "team": t.get("toTeam", {}).get("name")
                        or t.get("fromTeam", {}).get("name", "Unknown"),
                "type": t.get("typeDesc", "Unknown"),
                "date": t.get("date", ""),
                "description": t.get("description", ""),
            })

    return injuries


# ── Team Roster ───────────────────────────────────────────────────────────────

def get_team_id(team_name: str) -> Optional[int]:
    """
    Fuzzy-match a team name to its MLB team ID.
    """
    data = _get("/teams", params={"sportId": 1})
    for team in data.get("teams", []):
        if team_name.lower() in team["name"].lower():
            return team["id"]
    return None


def get_bullpen(team_id: int, season: int = 2025) -> list[dict]:
    """
    Returns all pitchers on a team's active roster with their recent stats.
    Useful for bullpen availability analysis.
    """
    data = _get(f"/teams/{team_id}/roster", params={
        "rosterType": "active",
        "season": season,
        "hydrate": "person(stats(group=pitching,type=season))"
    })

    pitchers = []
    for player in data.get("roster", []):
        pos = player.get("position", {}).get("type", "")
        if pos != "Pitcher":
            continue

        person = player.get("person", {})
        stats_list = person.get("stats", [])
        stats = {}
        if stats_list:
            splits = stats_list[0].get("splits", [{}])
            stats = splits[0].get("stat", {}) if splits else {}

        pitchers.append({
            "name": person.get("fullName", "Unknown"),
            "id": person.get("id"),
            "role": player.get("status", {}).get("description", "Active"),
            "era": stats.get("era", "N/A"),
            "games": stats.get("gamesPlayed", 0),
            "saves": stats.get("saves", 0),
            "strikeouts": stats.get("strikeOuts", 0),
            "whip": stats.get("whip", "N/A"),
        })

    return pitchers


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📅 TODAY'S GAMES")
    print("=" * 60)
    games = get_games()

    if not games:
        print("No games today. Trying yesterday...")
        games = get_games((date.today() - timedelta(days=1)).isoformat())

    for g in games:
        home_p = g['home_probable_pitcher']['name']
        away_p = g['away_probable_pitcher']['name']
        print(f"  {g['away_team']} @ {g['home_team']}")
        print(f"  Pitchers: {away_p} vs {home_p}")
        print(f"  Status: {g['status']} | Venue: {g['venue']}")
        print()

    # Test team ID lookup
    print("🏥 INJURY REPORT (last 7 days — Red Sox)")
    red_sox_id = get_team_id("Red Sox")
    print(f"  Red Sox team ID: {red_sox_id}")

    if red_sox_id:
        injuries = get_injuries(team_id=red_sox_id)
        if injuries:
            for inj in injuries[:5]:
                print(f"  • {inj['player']} ({inj['team']}) — {inj['type']} on {inj['date']}")
        else:
            print("  No recent IL transactions found.")

    # Test pitcher stats if we have a game
    if games:
        first_game = games[0]
        pitcher_id = first_game["home_probable_pitcher"]["id"]
        pitcher_name = first_game["home_probable_pitcher"]["name"]
        if pitcher_id:
            print(f"\n⚾ PITCHER STATS: {pitcher_name}")
            stats = get_pitcher_stats(pitcher_id)
            print(f"  ERA: {stats['era']} | WHIP: {stats['whip']} | K: {stats['strikeouts']}")
            print(f"  IP: {stats['innings_pitched']} | W-L: {stats['wins']}-{stats['losses']}")