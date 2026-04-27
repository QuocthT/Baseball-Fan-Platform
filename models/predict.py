"""
predict.py
----------
Baseline XGBoost Matchup Predictor for Baseball IQ.
Target Metric: Expected Pitcher Strikeouts (K)

Includes feature engineering for rolling wOBA, handedness splits, 
pitch values, and a 'Shadow Tracker' for backtesting predictions.
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pybaseball as pb

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_strikeout_model.json")
TRACKER_PATH = os.path.join(MODEL_DIR, "shadow_tracker.csv")

# Ensure cache is enabled for pybaseball to speed up repeated pulls
pb.cache.enable()


# ── Feature Engineering ───────────────────────────────────────────────────────

def calculate_rolling_woba(batter_id: int, target_date: str, days: int = 14) -> float:
    """
    Calculates a batter's rolling wOBA over the last N days prior to the target date.
    """
    end_date = pd.to_datetime(target_date)
    start_date = end_date - timedelta(days=days)
    
    try:
        # Pull statcast data for the date range
        df = pb.statcast_batter(start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'), 
                                player_id=batter_id)
        if df.empty:
            return 0.315  # League average wOBA fallback
        
        # Approximate wOBA from Statcast events (simplified linear weights)
        events = df['events'].value_counts()
        bb = events.get('walk', 0)
        hbp = events.get('hit_by_pitch', 0)
        singles = events.get('single', 0)
        doubles = events.get('double', 0)
        triples = events.get('triple', 0)
        hr = events.get('home_run', 0)
        
        # Denominator: AB + BB + SF + HBP (simplified)
        pa = len(df[df['events'].notna()])
        if pa == 0:
            return 0.315
            
        # Simplified wOBA formula weights
        woba = (0.69*bb + 0.72*hbp + 0.88*singles + 1.247*doubles + 1.578*triples + 2.031*hr) / pa
        return round(woba, 3)
    except Exception:
        return 0.315


def get_pitcher_split_metrics(pitcher_id: int, season: int) -> dict:
    """
    Calculates a pitcher's historical K% and whiff rates against LHB and RHB.
    """
    try:
        start = f"{season}-03-01"
        end = f"{season}-11-01"
        df = pb.statcast_pitcher(start, end, player_id=pitcher_id)
        
        if df.empty:
            return {"k_rate_vs_l": 0.22, "k_rate_vs_r": 0.22, "overall_whiff_rate": 0.25}

        # Calculate splits
        splits = {}
        for stand in ['L', 'R']:
            matchups = df[df['stand'] == stand]
            pa = len(matchups[matchups['events'].notna()])
            ks = len(matchups[matchups['events'] == 'strikeout'])
            splits[f"k_rate_vs_{stand.lower()}"] = ks / pa if pa > 0 else 0.22
            
        # Overall whiff rate (Pitch Value approximation)
        swings = df[df['description'].isin(['swinging_strike', 'foul', 'hit_into_play', 'foul_tip'])]
        whiffs = df[df['description'] == 'swinging_strike']
        splits["overall_whiff_rate"] = len(whiffs) / len(swings) if len(swings) > 0 else 0.25
        
        return splits
    except Exception:
         return {"k_rate_vs_l": 0.22, "k_rate_vs_r": 0.22, "overall_whiff_rate": 0.25}


def build_matchup_features(pitcher_id: int, lineup_ids: list[int], target_date: str) -> pd.DataFrame:
    """
    Combines pitcher historical stats and the lineup's rolling form into a single feature vector.
    """
    season = pd.to_datetime(target_date).year
    
    # 1. Pitcher features
    p_metrics = get_pitcher_split_metrics(pitcher_id, season)
    
    # 2. Lineup features (Average 14-day rolling wOBA of the starting 9)
    lineup_wobas = [calculate_rolling_woba(b_id, target_date) for b_id in lineup_ids[:9]]
    avg_lineup_woba = np.mean(lineup_wobas) if lineup_wobas else 0.315
    
    features = {
        "pitcher_k_rate_vs_l": p_metrics["k_rate_vs_l"],
        "pitcher_k_rate_vs_r": p_metrics["k_rate_vs_r"],
        "pitcher_whiff_rate": p_metrics["overall_whiff_rate"],
        "lineup_rolling_woba": avg_lineup_woba,
    }
    
    return pd.DataFrame([features])


# ── Model Training & Inference ────────────────────────────────────────────────

def train_baseline_model(training_data_path: str):
    """
    Trains the XGBoost model. 
    NOTE: You should build a historical CSV offline and pass it here.
    Expected CSV columns: [pitcher_k_rate_vs_l, pitcher_k_rate_vs_r, pitcher_whiff_rate, lineup_rolling_woba, target_strikeouts]
    """
    print("Loading training data...")
    df = pd.read_csv(training_data_path)
    
    features = ['pitcher_k_rate_vs_l', 'pitcher_k_rate_vs_r', 'pitcher_whiff_rate', 'lineup_rolling_woba']
    X = df[features]
    y = df['target_strikeouts']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Quick eval
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model trained! Validation MAE: {mae:.2f} Strikeouts")
    
    # Save model for Streamlit to use
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def predict_strikeouts(pitcher_id: int, lineup_ids: list[int], match_date: str = None) -> float:
    """
    Loads the saved XGBoost model and predicts strikeouts for a live matchup.
    """
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Returning baseline projection.")
        return 5.5 # Standard Vegas Over/Under baseline
        
    if match_date is None:
        match_date = date.today().isoformat()
        
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    X_pred = build_matchup_features(pitcher_id, lineup_ids, match_date)
    
    # Predict and ensure we don't return negative strikeouts
    prediction = max(0.0, model.predict(X_pred)[0])
    return round(prediction, 1)


# ── Shadow Tracker (Backtesting) ──────────────────────────────────────────────

def log_prediction(game_date: str, pitcher_name: str, pitcher_id: int, predicted_ks: float):
    """
    Saves the model's prediction to a local CSV file.
    """
    new_entry = pd.DataFrame([{
        "date": game_date,
        "pitcher_name": pitcher_name,
        "pitcher_id": pitcher_id,
        "predicted_ks": predicted_ks,
        "actual_ks": np.nan,  # To be filled in the next day
        "error": np.nan
    }])
    
    if os.path.exists(TRACKER_PATH):
        tracker_df = pd.read_csv(TRACKER_PATH)
        tracker_df = pd.concat([tracker_df, new_entry], ignore_index=True)
    else:
        tracker_df = new_entry
        
    tracker_df.to_csv(TRACKER_PATH, index=False)
    print(f"📝 Logged prediction for {pitcher_name}: {predicted_ks} Ks")


def evaluate_shadow_tracker():
    """
    Scans the shadow tracker for past games missing 'actual_ks', fetches the 
    actual box score data via pybaseball, and calculates the model's accuracy.
    """
    if not os.path.exists(TRACKER_PATH):
        print("No shadow tracker found.")
        return
        
    df = pd.read_csv(TRACKER_PATH)
    unresolved = df[df['actual_ks'].isna()]
    
    if unresolved.empty:
        print("✅ All past predictions have been graded. No pending evaluations.")
        
        # Print overall performance
        if not df['error'].isna().all():
            mae = df['error'].abs().mean()
            print(f"📊 Current Model MAE: {mae:.2f} Strikeouts off per game.")
        return

    print(f"🔍 Found {len(unresolved)} ungraded predictions. Fetching actuals...")
    
    for idx, row in unresolved.iterrows():
        try:
            # We fetch pitching stats for that specific date to get actual Ks
            # Note: pybaseball's statcast_pitcher allows date filtering
            stats = pb.statcast_pitcher(row['date'], row['date'], player_id=row['pitcher_id'])
            
            if stats.empty:
                continue # Game might have been rained out or pitcher didn't start
                
            actual_ks = len(stats[stats['events'] == 'strikeout'])
            
            df.at[idx, 'actual_ks'] = actual_ks
            df.at[idx, 'error'] = actual_ks - row['predicted_ks']
            print(f"  -> {row['pitcher_name']}: Predicted {row['predicted_ks']}, Actual {actual_ks}")
            
        except Exception as e:
            print(f"  -> Could not fetch actuals for {row['pitcher_name']}: {e}")
            
    df.to_csv(TRACKER_PATH, index=False)
    
    # Print updated performance
    graded = df.dropna(subset=['actual_ks'])
    if not graded.empty:
        mae = graded['error'].abs().mean()
        print(f"\n📈 Updated Model MAE: {mae:.2f} Strikeouts off per game.")


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 XGBoost Predictor & Shadow Tracker initialized.")
    print("To train a model, you must supply a historical CSV to `train_baseline_model()`.")
    
    # Example of how the evaluation loop works
    # evaluate_shadow_tracker()