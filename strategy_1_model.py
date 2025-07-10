# === strategy_1_model.py ===
import os
import joblib

# === Load Pre-trained Models ===
model_dir = "models"
buy_model_path = os.path.join(model_dir, "buy_model_latest.pkl")
rr_model_path = os.path.join(model_dir, "rr_model_latest.pkl")

if os.path.exists(buy_model_path) and os.path.exists(rr_model_path):
    buy_model = joblib.load(buy_model_path)
    rr_model = joblib.load(rr_model_path)
else:
    raise FileNotFoundError("❌ Pre-trained models not found. Please retrain via sidebar.")

# === RSI Computation for Live Predictions ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
