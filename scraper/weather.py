"""
weather.py
----------
Fetches live weather for MLB stadiums via OpenWeatherMap and maps 
it to historical Park Factors for the Pre-Game Brief.
"""

import os
import aiohttp
import streamlit as st
from typing import Optional

# ── Stadium Coordinates & Park Factors (wOBA index) ─────────────────────────
# Baseline is 1.00. > 1.00 is hitter-friendly, < 1.00 is pitcher-friendly.
MLB_STADIUMS = {
    "Arizona Diamondbacks": {"name": "Chase Field", "lat": 33.4455, "lon": -112.0667, "park_factor": 1.01},
    "Atlanta Braves": {"name": "Truist Park", "lat": 33.8907, "lon": -84.4677, "park_factor": 1.00},
    "Baltimore Orioles": {"name": "Oriole Park at Camden Yards", "lat": 39.2840, "lon": -76.6215, "park_factor": 1.00},
    "Boston Red Sox": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "park_factor": 1.03},
    "Chicago Cubs": {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "park_factor": 0.98},
    "Chicago White Sox": {"name": "Guaranteed Rate Field", "lat": 41.8300, "lon": -87.6338, "park_factor": 1.01},
    "Cincinnati Reds": {"name": "Great American Ball Park", "lat": 39.0979, "lon": -84.5072, "park_factor": 1.05},
    "Cleveland Guardians": {"name": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "park_factor": 1.00},
    "Colorado Rockies": {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942, "park_factor": 1.14},
    "Detroit Tigers": {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "park_factor": 1.01},
    "Houston Astros": {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555, "park_factor": 0.99},
    "Kansas City Royals": {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "park_factor": 1.01},
    "Los Angeles Angels": {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "park_factor": 0.99},
    "Los Angeles Dodgers": {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "park_factor": 1.01},
    "Miami Marlins": {"name": "loanDepot park", "lat": 25.7783, "lon": -80.2195, "park_factor": 0.98},
    "Milwaukee Brewers": {"name": "American Family Field", "lat": 43.0280, "lon": -87.9712, "park_factor": 0.99},
    "Minnesota Twins": {"name": "Target Field", "lat": 44.9817, "lon": -93.2778, "park_factor": 1.02},
    "New York Mets": {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458, "park_factor": 0.97},
    "New York Yankees": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "park_factor": 0.98},
    "Oakland Athletics": {"name": "Sutter Health Park", "lat": 38.5802, "lon": -121.5135, "park_factor": 1.02},
    "Philadelphia Phillies": {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "park_factor": 0.99},
    "Pittsburgh Pirates": {"name": "PNC Park", "lat": 40.4469, "lon": -80.0057, "park_factor": 1.00},
    "San Diego Padres": {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570, "park_factor": 0.98},
    "San Francisco Giants": {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "park_factor": 0.99},
    "Seattle Mariners": {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "park_factor": 0.96},
    "St. Louis Cardinals": {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "park_factor": 0.98},
    "Tampa Bay Rays": {"name": "Tropicana Field", "lat": 27.7682, "lon": -82.6534, "park_factor": 0.99},
    "Texas Rangers": {"name": "Globe Life Field", "lat": 32.7473, "lon": -97.0818, "park_factor": 0.96},
    "Toronto Blue Jays": {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "park_factor": 1.00},
    "Washington Nationals": {"name": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "park_factor": 0.98},
}

async def get_weather_and_park(_session: aiohttp.ClientSession, home_team: str) -> dict:
    stadium = MLB_STADIUMS.get(home_team)
    if not stadium:
        return {"error": "Stadium not found for team."}

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {
            "park_name": stadium["name"],
            "park_factor": stadium["park_factor"],
            "weather": "Weather data unavailable (add OPENWEATHER_API_KEY to .env)",
            "temp": "N/A",
            "wind": "N/A"
        }

    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": stadium["lat"],
        "lon": stadium["lon"],
        "appid": api_key,
        "units": "imperial" # Gets temperature in Fahrenheit
    }

    try:
        async with _session.get(url, params=params, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            
            # Identify if it's a dome (simplified logic)
            is_dome = stadium["name"] in ["Tropicana Field", "Minute Maid Park", "Globe Life Field", "loanDepot park", "American Family Field", "Chase Field", "Rogers Centre"]
            
            return {
                "park_name": stadium["name"],
                "park_factor": stadium["park_factor"],
                "is_dome": is_dome,
                "temp": f"{int(data['main']['temp'])}°F",
                "wind": f"{int(data['wind']['speed'])} mph",
                "description": data['weather'][0]['description'].capitalize(),
                "narrative": f"Game played at {stadium['name']} (Park Factor: {stadium['park_factor']}). Current weather is {int(data['main']['temp'])}°F with winds at {int(data['wind']['speed'])} mph." if not is_dome else f"Game played at {stadium['name']} (Park Factor: {stadium['park_factor']}). This is a dome/retractable roof stadium, so environmental weather is less impactful."
            }
    except Exception as e:
         return {
            "park_name": stadium["name"],
            "park_factor": stadium["park_factor"],
            "weather_error": str(e)
        }