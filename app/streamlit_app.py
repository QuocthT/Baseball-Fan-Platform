"""
streamlit_app.py
----------------
Baseball IQ — Full Pre-Game Intelligence Dashboard

Tabs:
  1. 🧠 Pre-Game Brief     — RAG-generated AI analysis
  2. ⚾ Pitcher Matchup    — Arsenal charts, velocity, whiff rates
  3. 👥 Lineup Analysis    — Batter profiles, hot/cold, weaknesses
  4. 🧑‍⚖️ Umpire Card       — Strike zone bias, accuracy visualization
  5. 📱 Fan Pulse          — Reddit posts + YouTube creator takes

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from datetime import date, datetime
import numpy as np
import time

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Baseball IQ",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Cards */
    .stat-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .stat-card h4 { color: #a0aec0; font-size: 12px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1px; }
    .stat-card p  { color: #f7fafc; font-size: 26px; font-weight: 700; margin: 0; }

    /* Section headers */
    .section-header {
        color: #63b3ed;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 6px;
    }

    /* Pitcher pill */
    .pitcher-pill {
        display: inline-block;
        background: #2b4a7a;
        color: #90cdf4;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px;
    }

    /* Hot/cold badges */
    .badge-hot  { background:#742a2a; color:#fc8181; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
    .badge-ok   { background:#1a4731; color:#68d391; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
    .badge-cold { background:#1a2d5a; color:#63b3ed; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }

    /* Brief output */
    .brief-box {
        background: #1a1f2e;
        border-left: 4px solid #63b3ed;
        border-radius: 0 12px 12px 0;
        padding: 24px;
        line-height: 1.8;
        color: #e2e8f0;
        font-size: 15px;
    }

    /* Reddit card */
    .reddit-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }
    .reddit-card .sub  { color:#f6993f; font-size:11px; font-weight:700; }
    .reddit-card .title { color:#e2e8f0; font-size:14px; font-weight:600; margin:4px 0; }
    .reddit-card .body  { color:#a0aec0; font-size:13px; }

    /* YouTube card */
    .yt-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-left: 3px solid #ff4444;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }
    .yt-card .channel { color:#fc8181; font-size:11px; font-weight:700; }
    .yt-card .title   { color:#e2e8f0; font-size:14px; font-weight:600; margin:4px 0; }
    .yt-card .summary { color:#a0aec0; font-size:13px; }

    /* Injury row */
    .inj-row {
        display:flex; align-items:center; gap:10px;
        background:#1a1f2e; border-radius:8px; padding:10px 14px; margin:4px 0;
    }

    /* Override streamlit defaults */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e;
        border-radius: 8px;
        color: #a0aec0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #2b4a7a !important;
        color: #90cdf4 !important;
    }
    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Imports ───────────────────────────────────────────────────────────────────

from scraper.mlb_api import get_games, get_pitcher_stats, get_lineup, get_injuries
from scraper.reddit_youtube import get_reddit_posts, get_youtube_summaries
from scraper.umpire import get_umpire_fallback

# ── Real game schedule ────────────────────────────────────────────────────────

real_games = get_games()

# Add a display label to each game
for g in real_games:
    g["label"] = f"{g['away_team']} @ {g['home_team']}"

# Fallback if no games today
if not real_games:
    st.error("No MLB games found for today. Try again later or check your connection.")
    st.stop()


# ── Data-fetching helpers (cached) ────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_pitcher(pitcher_id, pitcher_name: str) -> dict:
    """Fetch season stats for a pitcher; return UI-ready dict."""
    if not pitcher_id:
        return {
            "name": pitcher_name or "TBD",
            "era": "N/A", "whip": "N/A", "k9": "N/A",
            "wins": 0, "losses": 0, "ip": "N/A",
            "recent_form": "N/A",
            "scouting": "Probable pitcher has not been announced.",
            "arsenal": [],
        }

    stats = get_pitcher_stats(pitcher_id)
    era   = stats.get("era", "N/A")
    whip  = stats.get("whip", "N/A")
    k9    = stats.get("strikeout_per_9", "N/A")
    wins  = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    ip    = stats.get("innings_pitched", "N/A")

    scouting = (
        f"{pitcher_name} carries a {era} ERA and {whip} WHIP this season, "
        f"punching out batters at a {k9} K/9 clip across {ip} innings pitched."
        if era != "N/A" else
        f"Season stats for {pitcher_name} are not yet available."
    )

    return {
        "name": pitcher_name,
        "era": era,
        "whip": whip,
        "k9": k9,
        "wins": wins,
        "losses": losses,
        "ip": ip,
        "recent_form": "N/A",
        "scouting": scouting,
        "arsenal": [],   # Statcast arsenal requires pybaseball (see scraper/statcast.py)
    }


@st.cache_data(ttl=300)
def fetch_lineup(game_id: int) -> dict:
    """Fetch confirmed batting lineups; returns empty dict if not yet posted."""
    try:
        return get_lineup(game_id)
    except Exception:
        return {}


def _adapt_lineup(batting_order: list) -> list:
    """Convert MLB API batting_order list to the UI batter format."""
    adapted = []
    for b in batting_order:
        adapted.append({
            "name": b.get("name", "Unknown"),
            "pos":  b.get("position", "?"),
            "avg":  b.get("batting_avg", ".???"),
            "hr":   b.get("home_runs", 0),
            "ops":  b.get("ops", ".???"),
            "form": "✅ Normal",   # live form requires Statcast (scraper/statcast.py)
            "weakness": "N/A",    # weaknesses require Statcast
        })
    return adapted


@st.cache_data(ttl=600)
def fetch_injuries(home_team: str, away_team: str) -> list:
    """Pull IL transactions for the past 7 days and filter to tonight's teams."""
    try:
        raw = get_injuries(days_back=7)
    except Exception:
        return []

    team_words = {
        w.lower() for t in [home_team, away_team]
        for w in t.split()
        if len(w) > 3  # skip short words like "the"
    }

    injuries = []
    for t in raw:
        team_str = (t.get("team") or "").lower()
        if any(w in team_str for w in team_words):
            injuries.append({
                "player": t.get("player", "Unknown"),
                "team":   t.get("team", "Unknown"),
                "status": t.get("type", "IL"),
                "issue":  t.get("description", ""),
                "eta":    "TBD",
            })
    return injuries


@st.cache_data(ttl=300)
def fetch_fan_pulse(query: str):
    """Reddit posts + YouTube summaries for a matchup query."""
    reddit = get_reddit_posts(query, limit=6)
    youtube = get_youtube_summaries(query, max_results=4)

    # Normalize reddit field names (real API uses 'subreddit', mock uses 'sub')
    for p in reddit:
        if "subreddit" in p and "sub" not in p:
            p["sub"] = p["subreddit"]
        p.setdefault("sub", "r/baseball")
        p.setdefault("score", 0)
        p.setdefault("comments", 0)
        p.setdefault("body", "")

    # Normalize youtube field names
    for v in youtube:
        v.setdefault("views", "N/A")
        v.setdefault("summary", v.get("description", ""))

    return reddit, youtube


@st.cache_data(ttl=600)
def fetch_umpire(game_id: int) -> dict:
    """Try to get the home plate umpire from the boxscore; fallback to generic."""
    ump_name = "TBD"
    try:
        from scraper.mlb_api import _get
        data = _get(f"/game/{game_id}/boxscore")
        for official in data.get("officials", []):
            if official.get("officialType", "") == "Home Plate":
                ump_name = official.get("official", {}).get("fullName", "TBD")
                break
    except Exception:
        pass

    profile = get_umpire_fallback(ump_name) if ump_name != "TBD" else {}
    accuracy = float(profile.get("accuracy_pct", 92.8) or 92.8)
    narrative = profile.get("narrative", (
        f"Umpire assignment for this game has not yet been announced. "
        f"The 2025 MLB average ball/strike accuracy is 92.8% per Statcast. "
        f"In 2026, each team gets 2 ABS challenges per game."
    ))

    return {
        "name": ump_name,
        "accuracy": accuracy,
        "calls_per_game": 148,
        "missed_per_game": round(148 * (1 - accuracy / 100), 1),
        "favor": float(profile.get("favor_score", 0.0) or 0.0),
        "narrative": narrative,
        "zone_profile": {
            "inside_strike":  round(accuracy - 0.5, 1),
            "outside_strike": round(accuracy + 0.3, 1),
            "high_strike":    round(accuracy - 1.2, 1),
            "low_strike":     round(accuracy + 0.2, 1),
        },
    }


# ── Session State ─────────────────────────────────────────────────────────────

if "selected_game_idx" not in st.session_state:
    st.session_state.selected_game_idx = 0
if "brief_generated" not in st.session_state:
    st.session_state.brief_generated = False
if "use_ai" not in st.session_state:
    st.session_state.use_ai = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def form_badge(form: str) -> str:
    if "Hot" in form:   return f'<span class="badge-hot">{form}</span>'
    if "Cold" in form:  return f'<span class="badge-cold">{form}</span>'
    return f'<span class="badge-ok">{form}</span>'


def pitch_type_color(pitch: str) -> str:
    colors = {
        "4-Seam FB": "#f56565", "Sinker": "#4299e1", "Slider": "#ed8936",
        "Changeup": "#48bb78",  "Curve": "#9f7aea",  "Cutter": "#f6e05e",
        "Forkball": "#fc8181",  "Knuckle-CB": "#b794f4",
    }
    return colors.get(pitch, "#a0aec0")


def radar_chart(zone_profile: dict) -> go.Figure:
    """Strike zone accuracy radar chart."""
    labels = ["Inside Strike", "Outside Strike", "High Strike", "Low Strike"]
    values = [
        zone_profile["inside_strike"],
        zone_profile["outside_strike"],
        zone_profile["high_strike"],
        zone_profile["low_strike"],
    ]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed, fill="toself",
        fillcolor="rgba(99,179,237,0.2)", line=dict(color="#63b3ed", width=2),
        name="Accuracy %"
    ))
    avg = [93.0] * 4
    avg_closed = avg + [avg[0]]
    fig.add_trace(go.Scatterpolar(
        r=avg_closed, theta=labels_closed, fill="toself",
        fillcolor="rgba(160,174,192,0.05)",
        line=dict(color="#a0aec0", width=1, dash="dot"),
        name="MLB Avg (93%)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[85, 97], tickfont=dict(color="#a0aec0")),
            angularaxis=dict(tickfont=dict(color="#e2e8f0")),
            bgcolor="#1a1f2e",
        ),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(font=dict(color="#a0aec0")),
        margin=dict(l=40, r=40, t=20, b=20), height=320,
    )
    return fig


def arsenal_chart(arsenal: list, pitcher_name: str) -> go.Figure:
    """Grouped bar chart: pitch usage, velocity, whiff rate."""
    pitches = [p["pitch"] for p in arsenal]
    colors  = [p.get("color", "#63b3ed") for p in arsenal]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Usage %", "Avg Velocity (mph)", "Whiff Rate %"],
    )

    for i, (key, row) in enumerate([("pct", 1), ("velo", 2), ("whiff", 3)]):
        vals = [p[key] for p in arsenal]
        fig.add_trace(go.Bar(
            x=pitches, y=vals,
            marker_color=colors,
            showlegend=False,
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        ), row=1, col=row)

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font=dict(color="#a0aec0"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(tickfont=dict(color="#e2e8f0"))
    fig.update_yaxes(gridcolor="#2d3748", tickfont=dict(color="#a0aec0"))
    return fig

def plot_pitch_movement(df_pitches: pd.DataFrame) -> go.Figure:
    """Scatter plot of horizontal vs. vertical pitch break (pfx_x vs pfx_z)."""
    fig = px.scatter(
        df_pitches, x="pfx_x", y="pfx_z", color="pitch_type",
        color_discrete_map={
            "FF": "#f56565", "SI": "#4299e1", "SL": "#ed8936",
            "CH": "#48bb78", "CU": "#9f7aea", "FC": "#f6e05e", "ST": "#fc8181"
        },
        labels={"pfx_x": "Horizontal Break (in)", "pfx_z": "Vertical Break (in)", "pitch_type": "Pitch"}
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#4a5568")
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#4a5568")
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font=dict(color="#a0aec0"),
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_strike_zone(df_pitches: pd.DataFrame) -> go.Figure:
    """Heatmap of pitch locations over the plate (plate_x vs plate_z)."""
    fig = go.Figure()
    
    # 2D Density Contour
    fig.add_trace(go.Histogram2dContour(
        x=df_pitches["plate_x"], y=df_pitches["plate_z"],
        colorscale="YlOrRd", reversescale=False, showscale=False,
        ncontours=15, line=dict(width=0)
    ))
    
    # Draw the Strike Zone
    fig.add_shape(
        type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
        line=dict(color="white", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )
    
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font=dict(color="#a0aec0"), height=350,
        xaxis=dict(title="Plate X", range=[-2.5, 2.5], zeroline=False),
        yaxis=dict(title="Plate Z", range=[0, 5], zeroline=False),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def plot_spray_chart(df_hits: pd.DataFrame) -> go.Figure:
    """Basic batter spray chart using hit coordinates (hc_x, hc_y)."""
    fig = px.scatter(
        df_hits, x="hc_x", y="hc_y", color="events",
        color_discrete_map={"Single": "#63b3ed", "Double": "#48bb78", "Triple": "#ed8936", "Home Run": "#f56565"},
        labels={"hc_x": "", "hc_y": "", "events": "Result"}
    )
    # Mock baseball diamond overlay
    fig.add_shape(type="path", path="M 125 204 L 125 100 L 25 100 Z", line_color="#4a5568", fillcolor="rgba(0,0,0,0)")
    
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font=dict(color="#a0aec0"), height=300,
        xaxis=dict(range=[0, 250], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 250], showgrid=False, zeroline=False, visible=False, autorange="reversed"),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

# --- TEMPORARY MOCK DATA GENERATOR ---
def get_mock_statcast_pitcher(pitcher_name):
    """Generates fake pitch-by-pitch coordinate data until pybaseball is hooked up."""
    n_pitches = 200
    pitch_types = np.random.choice(["FF", "SL", "CH", "CU"], n_pitches, p=[0.5, 0.3, 0.15, 0.05])
    return pd.DataFrame({
        "pitch_type": pitch_types,
        "pfx_x": np.where(pitch_types == "FF", np.random.normal(-5, 2, n_pitches), np.random.normal(6, 3, n_pitches)),
        "pfx_z": np.where(pitch_types == "FF", np.random.normal(15, 2, n_pitches), np.random.normal(-2, 4, n_pitches)),
        "plate_x": np.random.normal(0, 1.2, n_pitches),
        "plate_z": np.random.normal(2.5, 1.0, n_pitches),
    })

def get_mock_spray_chart():
    """Generates fake batted ball coordinates."""
    n_hits = 50
    events = np.random.choice(["Single", "Double", "Home Run"], n_hits, p=[0.6, 0.3, 0.1])
    return pd.DataFrame({
        "events": events,
        "hc_x": np.random.normal(125, 40, n_hits),
        "hc_y": np.where(events == "Home Run", np.random.normal(40, 10, n_hits), np.random.normal(130, 30, n_hits))
    })

def try_real_rag(game: dict, home_p: dict, away_p: dict,
                 home_lineup: list, away_lineup: list,
                 injuries: list, umpire: dict,
                 reddit_posts: list, youtube: list) -> str:
    """Attempt to call real RAG pipeline."""
    try:
        from pipeline.rag import ingest_game_data, query_pregame_brief
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        game_data = {
            "game": game,
            "home_pitcher_stats": {
                "name": home_p["name"], "era": home_p["era"],
                "whip": home_p["whip"], "strikeouts": "N/A",
                "innings_pitched": home_p["ip"], "wins": home_p["wins"],
                "losses": home_p["losses"], "strikeout_per_9": home_p["k9"],
            },
            "away_pitcher_stats": {
                "name": away_p["name"], "era": away_p["era"],
                "whip": away_p["whip"], "strikeouts": "N/A",
                "innings_pitched": away_p["ip"], "wins": away_p["wins"],
                "losses": away_p["losses"], "strikeout_per_9": away_p["k9"],
            },
            "home_pitcher_arsenal": {
                "pitcher": home_p["name"],
                "arsenal": {p["pitch"]: {"usage_pct": p["pct"], "avg_velocity": p["velo"],
                                         "whiff_rate": p["whiff"]}
                            for p in home_p["arsenal"]},
            },
            "away_pitcher_arsenal": {
                "pitcher": away_p["name"],
                "arsenal": {p["pitch"]: {"usage_pct": p["pct"], "avg_velocity": p["velo"],
                                         "whiff_rate": p["whiff"]}
                            for p in away_p["arsenal"]},
            },
            "batter_profiles": [
                {"batter": b["name"], "weakness_summary": [b["weakness"]]}
                for b in away_lineup[:4]
            ],
            "injuries": [
                {"player": i["player"], "team": i["team"],
                 "type": i["status"], "date": game["date"],
                 "description": i["issue"]}
                for i in injuries
            ],
            "umpire": {
                "umpire": umpire["name"],
                "accuracy_pct": str(umpire["accuracy"]),
                "narrative": umpire["narrative"],
            },
            "reddit_posts": [
                {"subreddit": p["sub"], "title": p["title"], "body": p["body"]}
                for p in reddit_posts
            ],
            "youtube_summaries": [
                {"channel": y["channel"], "title": y["title"], "summary": y["summary"]}
                for y in youtube
            ],
        }

        n = ingest_game_data(game_data)
        matchup = f"{game['away_team']} vs {game['home_team']}"
        return query_pregame_brief(matchup, game_id=game["game_id"])

    except Exception as e:
        return f"⚠️ RAG pipeline error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚾ Baseball IQ")
    st.markdown(f"*{date.today().strftime('%A, %B %d, %Y')}*")
    st.divider()

    st.markdown("### 🗓️ Today's Games")
    game_labels = [g["label"] for g in real_games]
    selected_label = st.radio("Select a game:", game_labels, index=0)
    game = real_games[game_labels.index(selected_label)]

    st.divider()

    st.markdown("### ⚙️ AI Settings")
    use_real_ai = st.toggle("Use Real RAG Pipeline", value=False,
        help="Requires ANTHROPIC_API_KEY or GROQ_API_KEY in .env")

    if use_real_ai:
        st.info("🔑 Set your API key in `.env` file")

    st.divider()
    st.markdown("### 📡 Data Sources")
    st.markdown("""
- ✅ MLB Stats API
- ✅ Statcast / pybaseball
- ✅ UmpScorecards
- ⚙️ Reddit (add key)
- ⚙️ YouTube (add key)
    """)
    st.divider()
    st.caption("Baseball IQ v0.1 • Built with Streamlit + ChromaDB + Claude")


# ── Enrich selected game with real API data ───────────────────────────────────

home_prob = game["home_probable_pitcher"]
away_prob = game["away_probable_pitcher"]

home_p    = fetch_pitcher(home_prob["id"], home_prob["name"])
away_p    = fetch_pitcher(away_prob["id"], away_prob["name"])

raw_lineup   = fetch_lineup(game["game_id"])
home_lineup  = _adapt_lineup(raw_lineup.get("home", {}).get("batting_order", []))
away_lineup  = _adapt_lineup(raw_lineup.get("away", {}).get("batting_order", []))

injuries       = fetch_injuries(game["home_team"], game["away_team"])
umpire         = fetch_umpire(game["game_id"])
query          = f"{game['away_team']} {game['home_team']}"
reddit_posts, youtube = fetch_fan_pulse(query)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown(f"# {game['away_team']} @ {game['home_team']}")
    st.markdown(f"📍 {game['venue']}  &nbsp;&nbsp;  🗓️ {game['date']}  &nbsp;&nbsp;  📊 {game['status']}")
with col_meta:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:right">'
        f'<span class="pitcher-pill">{away_p["name"]}</span>'
        f' <span style="color:#a0aec0">vs</span> '
        f'<span class="pitcher-pill">{home_p["name"]}</span>'
        f'</div>', unsafe_allow_html=True
    )

st.divider()

# ── Top KPIs ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric(f"{away_p['name']}", f"ERA {away_p['era']}", f"WHIP {away_p['whip']}")
with k2: st.metric("K/9", away_p["k9"], f"Form: {away_p['recent_form']}")
with k3: st.metric(f"{home_p['name']}", f"ERA {home_p['era']}", f"WHIP {home_p['whip']}")
with k4: st.metric("K/9", home_p["k9"], f"Form: {home_p['recent_form']}")
with k5: st.metric("Umpire", umpire["name"], f"{umpire['accuracy']}% accuracy")
with k6:
    inj_count = len([i for i in injuries if "IL" in i["status"]])
    st.metric("IL'd Players", inj_count, "affecting tonight")

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Pre-Game Brief",
    "⚾ Pitcher Matchup",
    "👥 Lineup Analysis",
    "🧑‍⚖️ Umpire Card",
    "📱 Fan Pulse",
])


# ── TAB 1: Pre-Game Brief ─────────────────────────────────────────────────────
with tab1:
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        generate = st.button(
            "🧠 Generate AI Pre-Game Brief",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        if use_real_ai:
            st.info("🔗 Will call RAG pipeline → LLM (requires API key in .env)")
        else:
            st.info("📋 Toggle 'Use Real RAG Pipeline' in sidebar to generate an AI brief.")

    st.markdown("<br>", unsafe_allow_html=True)

    if generate:
        if use_real_ai:
            with st.spinner("📡 Scraping data → embedding → querying LLM..."):
                brief_text = try_real_rag(
                    game, home_p, away_p,
                    home_lineup, away_lineup,
                    injuries, umpire, reddit_posts, youtube,
                )
            st.session_state["brief_text"] = brief_text
            st.session_state["brief_generated"] = True
            st.markdown(
                f'<div class="brief-box">{brief_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Enable 'Use Real RAG Pipeline' in the sidebar to generate an AI brief.")

    elif st.session_state.get("brief_generated") and "brief_text" in st.session_state:
        st.markdown(
            f'<div class="brief-box">{st.session_state["brief_text"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="brief-box" style="opacity:0.5; text-align:center; padding: 60px;">
            <h3>👆 Click "Generate AI Pre-Game Brief" to get your analysis</h3>
            <p>Powered by RAG — pulls from MLB Stats API, injury reports, umpire data, Reddit & YouTube</p>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 2: Pitcher Matchup ────────────────────────────────────────────────────
with tab2:
    col_away_p, col_home_p = st.columns(2)

    for col, pitcher, side, opp_lineup in [
        (col_away_p, away_p, game["away_team"], home_lineup),
        (col_home_p, home_p, game["home_team"], away_lineup),
    ]:
        with col:
            st.markdown(f"### {pitcher['name']}")
            st.caption(f"{side} starter")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ERA",  pitcher["era"])
            m2.metric("WHIP", pitcher["whip"])
            m3.metric("K/9",  pitcher["k9"])
            m4.metric("W-L",  f"{pitcher['wins']}-{pitcher['losses']}")

            st.markdown(f"**Form:** {pitcher['recent_form']}")
            st.markdown(f"*{pitcher['scouting']}*")

            st.markdown('<p class="section-header">Arsenal & Movement</p>', unsafe_allow_html=True)
            
            # Use real arsenal if available, otherwise fallback
            if pitcher["arsenal"]:
                st.plotly_chart(
                    arsenal_chart(pitcher["arsenal"], pitcher["name"]),
                    use_container_width=True, key=f"arsenal_{side}"
                )
            
            # Advanced Statcast Visuals
            df_statcast = get_mock_statcast_pitcher(pitcher["name"])
            
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.markdown("<div style='text-align:center; color:#a0aec0; font-size:12px;'>Pitch Movement (Break)</div>", unsafe_allow_html=True)
                st.plotly_chart(plot_pitch_movement(df_statcast), use_container_width=True, key=f"move_{side}")
            with plot_col2:
                st.markdown("<div style='text-align:center; color:#a0aec0; font-size:12px;'>Pitch Location Heatmap</div>", unsafe_allow_html=True)
                st.plotly_chart(plot_strike_zone(df_statcast), use_container_width=True, key=f"zone_{side}")

            st.markdown('<p class="section-header">Lineup Threats vs This Pitcher</p>',
                       unsafe_allow_html=True)
            if opp_lineup:
                hot_batters = [b for b in opp_lineup if "Hot" in b["form"]]
                if hot_batters:
                    for b in hot_batters:
                        st.markdown(
                            f"🔥 **{b['name']}** ({b['pos']}) — "
                            f"{b['avg']} AVG, {b['hr']} HR  "
                            f"| Weakness: *{b['weakness']}*"
                        )
                else:
                    st.markdown("*No confirmed hot hitters in opposing lineup tonight.*")
            else:
                st.info("Lineup not yet posted (typically released ~1 hr before first pitch).")

    # Head-to-head highlights
    st.divider()
    st.markdown("### 🔁 Pitcher Highlights")
    hth_cols = st.columns(2)
    with hth_cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <h4>{away_p['name'].split()[0]} Season</h4>
            <p style="font-size:16px">{away_p['era']} ERA · {away_p['whip']} WHIP · {away_p['k9']} K/9</p>
        </div>
        """, unsafe_allow_html=True)
    with hth_cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <h4>{home_p['name'].split()[0]} Season</h4>
            <p style="font-size:16px">{home_p['era']} ERA · {home_p['whip']} WHIP · {home_p['k9']} K/9</p>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 3: Lineup Analysis ────────────────────────────────────────────────────
with tab3:
    col_away_l, col_home_l = st.columns(2)

    for col, lineup, team_name, opp_pitcher in [
        (col_away_l, away_lineup, game["away_team"], home_p),
        (col_home_l, home_lineup, game["home_team"], away_p),
    ]:
        with col:
            st.markdown(f"### {team_name} Lineup")
            st.caption(f"Facing: {opp_pitcher['name']} ({opp_pitcher['era']} ERA)")

            if not lineup:
                st.info(
                    "📋 Lineup not yet posted. "
                    "Check back closer to first pitch (~1 hour before game time)."
                )
                continue

            hot  = sum(1 for b in lineup if "Hot" in b["form"])
            cold = sum(1 for b in lineup if "Cold" in b["form"])
            ok   = len(lineup) - hot - cold
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("🔥 Hot", hot)
            hc2.metric("✅ Normal", ok)
            hc3.metric("❄️ Cold", cold)

            st.markdown("<br>", unsafe_allow_html=True)

            for i, b in enumerate(lineup, 1):
                badge = form_badge(b["form"])
                with st.container():
                    bcol1, bcol2, bcol3, bcol4 = st.columns([3, 1.5, 1.5, 3])
                    with bcol1:
                        st.markdown(
                            f"**{i}. {b['name']}** <span style='color:#a0aec0;font-size:12px'>{b['pos']}</span>",
                            unsafe_allow_html=True
                        )
                    with bcol2:
                        st.markdown(f"`{b['avg']}` avg")
                    with bcol3:
                        st.markdown(f"`{b['hr']}` HR")
                    with bcol4:
                        st.markdown(
                            badge + f" &nbsp; <span style='color:#718096;font-size:12px'>⚠️ {b['weakness']}</span>",
                            unsafe_allow_html=True
                        )
                if i < len(lineup):
                    st.markdown('<hr style="margin:4px 0;border-color:#2d3748">', unsafe_allow_html=True)

            # Batter Spray Chart (Team Level)
            st.markdown('<p class="section-header">Team Spray Chart (Recent 14 Days)</p>', unsafe_allow_html=True)
            df_spray = get_mock_spray_chart()
            st.plotly_chart(plot_spray_chart(df_spray), use_container_width=True, key=f"spray_{team_name}")

            # OPS chart
            st.markdown('<p class="section-header">OPS by Batting Position</p>', unsafe_allow_html=True)
            try:
                ops_vals = [float(b["ops"]) for b in lineup]
                ops_colors = [
                    "#f56565" if "Hot" in b["form"] else
                    "#63b3ed" if "Cold" in b["form"] else "#48bb78"
                    for b in lineup
                ]
                fig_ops = go.Figure(go.Bar(
                    x=[f"{i+1}. {b['name'].split()[0]}" for i, b in enumerate(lineup)],
                    y=ops_vals,
                    marker_color=ops_colors,
                    text=[b["ops"] for b in lineup],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=10),
                ))
                fig_ops.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                    font=dict(color="#a0aec0"), height=220,
                    margin=dict(l=10, r=10, t=10, b=40),
                    yaxis=dict(range=[0.5, 1.1], gridcolor="#2d3748"),
                    xaxis=dict(tickfont=dict(size=9)),
                )
                st.plotly_chart(fig_ops, use_container_width=True, key=f"ops_{team_name}")
            except (ValueError, TypeError):
                st.caption("OPS chart unavailable — season stats may not yet be populated.")


# ── TAB 4: Umpire Card ────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"## 🧑‍⚖️ Tonight's Umpire: {umpire['name']}")

    ucol1, ucol2 = st.columns([1, 1])

    with ucol1:
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Accuracy", f"{umpire['accuracy']}%",
                  "Above avg" if umpire['accuracy'] >= 92.8 else "⚠️ Below avg")
        u2.metric("Calls/Game", umpire["calls_per_game"])
        u3.metric("Missed/Game", umpire["missed_per_game"],
                  f"-{round(148 - umpire['calls_per_game'] * (umpire['accuracy']/100), 1)} vs avg")
        favor_label = f"{'Home' if umpire['favor'] > 0 else 'Away'} lean"
        u4.metric("Favor Score", umpire["favor"], favor_label)

        st.markdown('<p class="section-header">Umpire Scouting Report</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="brief-box">
            {umpire['narrative']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">ABS Challenge Reminder</p>', unsafe_allow_html=True)
        weakest = min(umpire["zone_profile"], key=umpire["zone_profile"].get)
        st.info(
            "⚖️ **2026 Rule:** Each team gets **2 ABS challenges** per 9 innings. "
            "Retain the challenge if the umpire is overturned. "
            f"Given {umpire['name']}'s zone profile, prioritize challenges on "
            f"**{weakest.replace('_', ' ')}** pitches."
        )

    with ucol2:
        st.markdown('<p class="section-header">Zone Accuracy by Location</p>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(umpire["zone_profile"]), use_container_width=True, key="umpire_radar")

        st.markdown('<p class="section-header">Zone Breakdown</p>', unsafe_allow_html=True)
        zone_df = pd.DataFrame([
            {"Zone": k.replace("_", " ").title(), "Accuracy %": v}
            for k, v in umpire["zone_profile"].items()
        ])
        fig_zone = px.bar(
            zone_df, x="Zone", y="Accuracy %",
            color="Accuracy %",
            color_continuous_scale=["#f56565", "#ed8936", "#48bb78"],
            range_color=[88, 96],
            text="Accuracy %",
        )
        fig_zone.add_hline(y=92.8, line_dash="dot", line_color="#a0aec0",
                           annotation_text="MLB Avg", annotation_font_color="#a0aec0")
        fig_zone.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_zone.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
            font=dict(color="#a0aec0"), height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            yaxis=dict(range=[85, 97], gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_zone, use_container_width=True, key="zone_bar")

    st.divider()
    st.markdown("### 🎯 Strategic Implications for Both Teams")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        **For {game['away_team']} ({away_p['name']}):**
        - Weakest zone: **{weakest.replace('_', ' ').title()}** at {umpire['zone_profile'][weakest]}%
        - Expand the zone early in counts to work ahead
        - Challenge tip: Save challenges for high-leverage 2-strike counts
        """)
    with sc2:
        st.markdown(f"""
        **For {game['home_team']} ({home_p['name']}):**
        - Framing catchers have an edge tonight
        - Attack the zone early in counts — don't let hitters work deep
        - Challenge tip: Prioritize inside pitches if accuracy is below 90%
        """)


# ── TAB 5: Fan Pulse ─────────────────────────────────────────────────────────
with tab5:
    fp1, fp2 = st.columns([1, 1])

    with fp1:
        st.markdown("### 📱 Reddit Buzz")
        st.caption("Latest posts from r/baseball, r/mlb + team subreddits")

        for post in reddit_posts:
            st.markdown(f"""
            <div class="reddit-card">
                <div class="sub">{post['sub']} &nbsp;·&nbsp; 🔺 {post['score']} &nbsp;·&nbsp; 💬 {post['comments']}</div>
                <div class="title">{post['title']}</div>
                <div class="body">{post['body']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏥 Injury Report")
        if injuries:
            for inj in injuries:
                status_color = "#fc8181" if "IL" in inj["status"] else "#f6e05e" if "Day" in inj["status"] else "#68d391"
                st.markdown(f"""
                <div class="inj-row">
                    <span style="color:{status_color};font-weight:700;font-size:12px;min-width:90px">{inj['status']}</span>
                    <span style="color:#e2e8f0;font-weight:600">{inj['player']}</span>
                    <span style="color:#a0aec0;font-size:12px">({inj['team']})</span>
                    <span style="color:#718096;font-size:12px">— {inj['issue']}</span>
                    <span style="color:#63b3ed;font-size:12px;margin-left:auto">ETA: {inj['eta']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent IL transactions found for tonight's teams.")

    with fp2:
        st.markdown("### 📺 Creator Takes")
        st.caption("Jomboy Media, Foolish Baseball, Pitching Ninja + more")

        for yt in youtube:
            views_label = f"👁 {yt['views']} views" if yt.get("views") not in (None, "N/A") else ""
            st.markdown(f"""
            <div class="yt-card">
                <div class="channel">▶ {yt['channel']} &nbsp;·&nbsp; {views_label}</div>
                <div class="title">{yt['title']}</div>
                <div class="summary">{yt['summary']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sentiment gauge from Reddit
        if reddit_posts:
            st.markdown("### 📊 Fan Sentiment")
            home_word = game["home_team"].split()[-1].lower()
            away_word = game["away_team"].split()[-1].lower()
            home_mentions = sum(
                1 for p in reddit_posts
                if home_word in p["title"].lower() or home_word in p["body"].lower()
            )
            away_mentions = len(reddit_posts) - home_mentions

            fig_sent = go.Figure(go.Bar(
                x=[game["away_team"].split()[-1], game["home_team"].split()[-1]],
                y=[away_mentions, home_mentions],
                marker_color=["#63b3ed", "#f56565"],
                text=[f"{away_mentions} posts", f"{home_mentions} posts"],
                textposition="outside",
            ))
            fig_sent.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                font=dict(color="#a0aec0"), height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(gridcolor="#2d3748"),
                title=dict(text="Reddit Post Mentions", font=dict(color="#a0aec0", size=12)),
            )
            st.plotly_chart(fig_sent, use_container_width=True, key="sentiment")

        st.markdown('<p class="section-header">Connect Live Data</p>', unsafe_allow_html=True)
        st.markdown("""
        To pull real Reddit & YouTube data:
        ```bash
        # Reddit — free at reddit.com/prefs/apps
        REDDIT_CLIENT_ID=your_id
        REDDIT_CLIENT_SECRET=your_secret

        # YouTube — free at console.cloud.google.com
        YOUTUBE_API_KEY=your_key
        ```
        Add these to your `.env` file and the scrapers activate automatically.
        """)
