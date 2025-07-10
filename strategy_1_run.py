import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import joblib
import numpy as np
import os
import subprocess
from functools import lru_cache

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")

# === Groww API Auth ===
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# Set token for downstream use
os.environ["GROWW_API_AUTH_TOKEN"] = api_key

# === Optional: Trigger Retraining ===
if st.sidebar.button("🧠 Retrain Model"):
    with st.spinner("Retraining in progress..."):
        result = subprocess.run(["python", "train_strategy_1.py"], capture_output=True, text=True)
        st.code(result.stdout + "\n" + result.stderr)
        if result.returncode == 0:
            st.success("✅ Retraining completed successfully.")
        else:
            st.error("❌ Retraining failed.")
    st.stop()

# === Initialize Groww API ===
groww = GrowwAPI(api_key)

# === Cached Instrument Metadata ===
@lru_cache(maxsize=1)
def load_instruments():
    df = pd.read_csv("instruments.csv")
    groww.instruments = df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: df
    groww.get_instrument_by_groww_symbol = lambda symbol: df[df['groww_symbol'] == symbol].iloc[0].to_dict()
    return df

instruments_df = load_instruments()

# === Load Models ===
try:
    from strategy_1_model import buy_model, rr_model, compute_rsi
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# === Set Date Range ===
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# === Auto Refresh (only in trading hours) ===
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)  # Refresh every 10 mins

# === Strategy Comparison Dropdown ===
strategy_option = st.sidebar.selectbox("Select Strategy Version", ["Strategy 1"])

# === Main Prediction Function ===
def live_predict(symbol="NSE-NIFTY", interval_minutes=10):
    start_str = start_time_ist.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time_ist.strftime("%Y-%m-%d %H:%M:%S")

    selected = groww.get_instrument_by_groww_symbol(symbol)

    data = groww.get_historical_candle_data(
        trading_symbol=selected['trading_symbol'],
        exchange=selected['exchange'],
        segment=selected['segment'],
        start_time=start_str,
        end_time=end_str,
        interval_in_minutes=interval_minutes
    )

    if isinstance(data, dict) and 'candles' in data and len(data['candles']) > 0:
        df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Add Features
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['Volatility'] = df['close'].rolling(window=10).std()
        df['RSI'] = compute_rsi(df['close'])

        latest = df[['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']].dropna().tail(1)

        if latest.empty:
            st.warning("Not enough data to predict.")
            return

        proba = buy_model.predict_proba(latest)[0]
        buy_signal = int(proba[1] > 0.5)
        rr_signal = rr_model.predict(latest)[0]

        st.metric("🕒 Last Candle", df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"))
        st.metric("📈 Signal", "BUY" if buy_signal == 1 else "HOLD / SELL")
        st.metric("🎯 Confidence", f"{proba[1]*100:.2f}%")
        st.metric("📊 Risk/Reward", f"{rr_signal:.4f}")
        st.dataframe(df.tail(10), use_container_width=True)

        # Trade Entry Timer
        next_candle_time = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)
        remaining = next_candle_time - datetime.now(ZoneInfo("Asia/Kolkata"))
        st.info(f"⏳ Time until next candle: {remaining.seconds // 60}m {remaining.seconds % 60}s")
    else:
        st.error("⚠️ No candle data returned from Groww API.")

# === Run Live Prediction ===
live_predict()

# === Manual Refresh ===
if st.button("🔁 Refresh Now"):
    st.rerun()
