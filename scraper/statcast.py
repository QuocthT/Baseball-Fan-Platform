"""
statcast.py
-----------
Pulls Statcast pitch-level data using pybaseball (no API key needed).
Covers: pitcher arsenal/tendencies, batter heatmaps, matchup history,
        recent performance, and spray chart data.

Run locally: pip install pybaseball
"""

import pandas as pd
from datetime import date, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import pybaseball as pb
    pb.cache.enable()  # cache requests so you don't re-download
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("⚠️  pybaseball not installed. Run: pip install pybaseball")


# ── Pitcher Arsenal ───────────────────────────────────────────────────────────

def get_pitcher_arsenal(pitcher_name: str, season: int = 2025) -> dict:
    """
    Returns a pitcher's pitch mix, velocity, spin rate, and effectiveness
    broken down by pitch type.

    Example output:
    {
      "FF": {"usage_pct": 55.2, "avg_velocity": 94.3, "whiff_rate": 28.1, ...},
      "SL": {"usage_pct": 28.1, "avg_velocity": 86.7, "whiff_rate": 34.5, ...},
      ...
    }
    """
    if not PYBASEBALL_AVAILABLE:
        return {}

    # Look up pitcher ID
    player = pb.playerid_lookup(
        pitcher_name.split()[-1],   # last name
        pitcher_name.split()[0]     # first name
    )
    if player.empty:
        return {"error": f"Could not find player: {pitcher_name}"}

    pitcher_id = int(player.iloc[0]["key_mlbam"])

    # Pull statcast data for the season
    start = f"{season}-03-01"
    end = f"{season}-11-01"

    df = pb.statcast_pitcher(start, end, player_id=pitcher_id)
    if df.empty:
        return {"error": "No data found"}

    df = df[df["pitch_type"].notna()]
    total_pitches = len(df)

    arsenal = {}
    for pitch_type, group in df.groupby("pitch_type"):
        n = len(group)
        swings = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked",
            "foul", "hit_into_play", "foul_tip"
        ])]
        whiffs = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul_tip"
        ])]

        arsenal[pitch_type] = {
            "usage_pct": round(n / total_pitches * 100, 1),
            "count": n,
            "avg_velocity": round(group["release_speed"].mean(), 1),
            "avg_spin_rate": round(group["release_spin_rate"].mean(), 0),
            "avg_extension": round(group["release_extension"].mean(), 2),
            "whiff_rate": round(len(whiffs) / len(swings) * 100, 1) if len(swings) > 0 else 0.0,
            "avg_horizontal_break": round(group["pfx_x"].mean() * 12, 1),  # convert to inches
            "avg_vertical_break": round(group["pfx_z"].mean() * 12, 1),
        }

    return {
        "pitcher": pitcher_name,
        "pitcher_id": pitcher_id,
        "season": season,
        "total_pitches": total_pitches,
        "arsenal": arsenal,
    }


# ── Batter Tendencies & Weaknesses ────────────────────────────────────────────

def get_batter_profile(batter_name: str, season: int = 2025) -> dict:
    """
    Returns a batter's strengths/weaknesses by pitch type and location.
    Key metrics: zone batting avg, chase rate, contact rate, hard hit %.
    """
    if not PYBASEBALL_AVAILABLE:
        return {}

    player = pb.playerid_lookup(
        batter_name.split()[-1],
        batter_name.split()[0]
    )
    if player.empty:
        return {"error": f"Could not find player: {batter_name}"}

    batter_id = int(player.iloc[0]["key_mlbam"])

    start = f"{season}-03-01"
    end = f"{season}-11-01"

    df = pb.statcast_batter(start, end, player_id=batter_id)
    if df.empty:
        return {"error": "No data found"}

    df = df[df["pitch_type"].notna()]

    # Overall stats
    in_zone = df[df["zone"].between(1, 9)]
    out_zone = df[df["zone"].between(11, 14)]

    chases = out_zone[out_zone["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul",
        "hit_into_play", "foul_tip"
    ])]
    chase_rate = len(chases) / len(out_zone) * 100 if len(out_zone) > 0 else 0

    contact = df[df["description"].isin(["foul", "hit_into_play", "foul_tip"])]
    swings = df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul",
        "hit_into_play", "foul_tip"
    ])]
    contact_rate = len(contact) / len(swings) * 100 if len(swings) > 0 else 0

    hard_hits = df[df["launch_speed"] >= 95]
    hard_hit_pct = len(hard_hits) / len(df[df["launch_speed"].notna()]) * 100 \
        if len(df[df["launch_speed"].notna()]) > 0 else 0

    # By pitch type breakdown
    pitch_breakdown = {}
    for pitch_type, group in df.groupby("pitch_type"):
        g_swings = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "hit_into_play", "foul_tip"
        ])]
        g_whiffs = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked"
        ])]
        g_hits = group[group["events"].isin([
            "single", "double", "triple", "home_run"
        ])]
        g_abs = group[group["events"].notna() & ~group["events"].isin([
            "walk", "hit_by_pitch", "sac_fly", "sac_bunt", "intent_walk"
        ])]

        pitch_breakdown[pitch_type] = {
            "pitches_seen": len(group),
            "avg_velocity_seen": round(group["release_speed"].mean(), 1),
            "whiff_rate": round(len(g_whiffs) / len(g_swings) * 100, 1) if g_swings.shape[0] > 0 else 0,
            "batting_avg_against": round(len(g_hits) / len(g_abs), 3) if g_abs.shape[0] > 0 else 0,
            "avg_exit_velocity": round(group["launch_speed"].mean(), 1) if group["launch_speed"].notna().any() else None,
        }

    return {
        "batter": batter_name,
        "batter_id": batter_id,
        "season": season,
        "overall": {
            "chase_rate": round(chase_rate, 1),
            "contact_rate": round(contact_rate, 1),
            "hard_hit_pct": round(hard_hit_pct, 1),
            "avg_exit_velocity": round(df["launch_speed"].mean(), 1) if df["launch_speed"].notna().any() else None,
            "avg_launch_angle": round(df["launch_angle"].mean(), 1) if df["launch_angle"].notna().any() else None,
        },
        "vs_pitch_types": pitch_breakdown,
        "weakness_summary": _summarize_weaknesses(pitch_breakdown),
    }


def _summarize_weaknesses(pitch_breakdown: dict) -> list[str]:
    """
    Auto-generates human-readable weakness bullets from pitch breakdown.
    These feed directly into the RAG context as natural language.
    """
    weaknesses = []
    for pitch, stats in pitch_breakdown.items():
        if stats["whiff_rate"] > 30 and stats["pitches_seen"] >= 20:
            weaknesses.append(
                f"Struggles vs {pitch} — {stats['whiff_rate']}% whiff rate "
                f"on {stats['pitches_seen']} pitches seen"
            )
        if stats.get("batting_avg_against", 1) < 0.180 and stats["pitches_seen"] >= 20:
            weaknesses.append(
                f"Hits only .{int(stats['batting_avg_against']*1000):03d} "
                f"against {pitch}"
            )
    return weaknesses


# ── Matchup History ───────────────────────────────────────────────────────────

def get_head_to_head(batter_name: str, pitcher_name: str,
                     seasons: list[int] = [2023, 2024, 2025]) -> dict:
    """
    Returns historical batter vs. pitcher matchup data across multiple seasons.
    This is limited by sample size but gives real situational context.
    """
    if not PYBASEBALL_AVAILABLE:
        return {}

    b_player = pb.playerid_lookup(batter_name.split()[-1], batter_name.split()[0])
    p_player = pb.playerid_lookup(pitcher_name.split()[-1], pitcher_name.split()[0])

    if b_player.empty or p_player.empty:
        return {"error": "Could not find one or both players"}

    b_id = int(b_player.iloc[0]["key_mlbam"])
    p_id = int(p_player.iloc[0]["key_mlbam"])

    all_data = []
    for season in seasons:
        try:
            df = pb.statcast_batter(f"{season}-03-01", f"{season}-11-01", player_id=b_id)
            matchup = df[df["pitcher"] == p_id]
            if not matchup.empty:
                all_data.append(matchup)
        except Exception:
            continue

    if not all_data:
        return {
            "batter": batter_name,
            "pitcher": pitcher_name,
            "note": "No historical matchup data found (may have never faced each other)",
            "pa": 0,
        }

    combined = pd.concat(all_data)
    pa = combined[combined["events"].notna() & ~combined["events"].isin([
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt", "intent_walk"
    ])]
    hits = combined[combined["events"].isin(["single", "double", "triple", "home_run"])]
    hrs = combined[combined["events"] == "home_run"]
    ks = combined[combined["events"] == "strikeout"]

    return {
        "batter": batter_name,
        "pitcher": pitcher_name,
        "pa": len(pa),
        "hits": len(hits),
        "home_runs": len(hrs),
        "strikeouts": len(ks),
        "batting_avg": round(len(hits) / len(pa), 3) if len(pa) > 0 else None,
        "note": f"Data from seasons: {seasons}",
    }


# ── Recent Form ───────────────────────────────────────────────────────────────

def get_recent_batter_form(batter_name: str, days: int = 14) -> dict:
    """
    Returns a batter's last N days of performance — the "hot/cold" signal.
    """
    if not PYBASEBALL_AVAILABLE:
        return {}

    player = pb.playerid_lookup(batter_name.split()[-1], batter_name.split()[0])
    if player.empty:
        return {"error": f"Player not found: {batter_name}"}

    batter_id = int(player.iloc[0]["key_mlbam"])
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()

    df = pb.statcast_batter(start, end, player_id=batter_id)
    if df.empty:
        return {"batter": batter_name, "note": "No recent data", "days": days}

    hits = df[df["events"].isin(["single", "double", "triple", "home_run"])]
    abs_ = df[df["events"].notna() & ~df["events"].isin([
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt", "intent_walk"
    ])]

    return {
        "batter": batter_name,
        "days": days,
        "pa": len(abs_),
        "hits": len(hits),
        "home_runs": len(df[df["events"] == "home_run"]),
        "batting_avg": round(len(hits) / len(abs_), 3) if len(abs_) > 0 else None,
        "avg_exit_velocity": round(df["launch_speed"].mean(), 1) if df["launch_speed"].notna().any() else None,
        "hard_hit_pct": round(
            len(df[df["launch_speed"] >= 95]) / len(df[df["launch_speed"].notna()]) * 100, 1
        ) if df["launch_speed"].notna().any() else None,
        "form": _classify_form(len(hits), len(abs_)),
    }


def _classify_form(hits: int, abs_: int) -> str:
    if abs_ == 0:
        return "No data"
    avg = hits / abs_
    if avg >= 0.320:
        return "🔥 Hot"
    elif avg >= 0.250:
        return "✅ Normal"
    elif avg >= 0.180:
        return "❄️ Cold"
    else:
        return "🧊 Ice Cold"


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("This module requires network access.")
    print("Run locally with: python scraper/statcast.py")
    print()
    print("Example usage:")
    print("  from scraper.statcast import get_pitcher_arsenal, get_batter_profile")
    print("  arsenal = get_pitcher_arsenal('Gerrit Cole')")
    print("  profile = get_batter_profile('Rafael Devers')")
    print("  h2h = get_head_to_head('Rafael Devers', 'Gerrit Cole')")