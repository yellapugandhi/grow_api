import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import joblib
import numpy as np
from functools import lru_cache
import subprocess
import sys
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")

# === Groww API Auth ===
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

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

# === Auto Refresh (during trading hours) ===
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)

# === Strategy Dropdown ===
strategy_option = st.sidebar.selectbox("Select Strategy Version", ["Strategy 1"])

# === Instrument Selection ===
symbol = st.sidebar.selectbox("Select Instrument", instruments_df['groww_symbol'].unique())

# === Live Prediction Function ===
def live_predict(symbol=symbol, interval_minutes=10):
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

        # Feature Engineering
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
        confidence = proba[1] * 100
        buy_signal = int(proba[1] > 0.5)
        rr_signal = rr_model.predict(latest)[0]

        st.markdown("### 🕒 Last Candle")
        st.markdown(f"**{df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}**")

        st.markdown("### 📈 Signal")
        st.markdown(f"**{'BUY' if buy_signal == 1 else 'HOLD / SELL'}**")

        # Confidence Strength Label
        if confidence >= 90:
            strength = "🔥 Strong BUY"
            color = "green"
        elif confidence >= 70:
            strength = "✅ Moderate BUY"
            color = "green"
        elif confidence >= 50:
            strength = "⚠️ Weak BUY"
            color = "orange"
        elif confidence >= 30:
            strength = "⚠️ Weak SELL"
            color = "orange"
        elif confidence >= 10:
            strength = "❌ Moderate SELL"
            color = "red"
        else:
            strength = "💀 Strong SELL"
            color = "darkred"

        st.markdown(
            f"### 🎯 <b>Confidence:</b> <span style='color:{color}'>{confidence:.2f}% - {strength}</span>",
            unsafe_allow_html=True
        )

        st.markdown(f"### 📊 <b>Risk/Reward:</b> `{rr_signal:.4f}`", unsafe_allow_html=True)

        st.dataframe(df.tail(10), use_container_width=True)

        # Real-time Chart
        st.line_chart(df.set_index('timestamp')['close'])

        # Feature Importance (SHAP)
        explainer = shap.Explainer(buy_model)
        shap_values = explainer(latest)
        st.subheader("🧠 Feature Importance")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        # Candle Timer
        next_candle_time = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)
        remaining = next_candle_time - datetime.now(ZoneInfo("Asia/Kolkata"))
        st.info(f"⏳ Time until next candle: {remaining.seconds // 60}m {remaining.seconds % 60}s")

        # Store to logs (optional extension)
        with open("prediction_logs.csv", "a") as log_file:
            log_file.write(f"{datetime.now()}, {symbol}, {confidence:.2f}, {rr_signal:.4f}, {'BUY' if buy_signal else 'SELL'}\n")

    else:
        st.error("⚠️ No candle data returned from Groww API.")

# === Predict Now ===
live_predict()

# === Retrain Trigger ===
st.sidebar.markdown("### 🧠 Retrain")
if st.sidebar.button("🔁 Retrain Model"):
    with st.sidebar:
        st.info("📡 Starting retraining...")
        try:
            process = subprocess.Popen(
                [sys.executable, "strategy_1_retrain.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            output_area = st.empty()
            logs = ""

            for line in process.stdout:
                logs += line
                output_area.code(logs)

            process.wait()

            if process.returncode == 0:
                st.success("✅ Retraining complete!")
                st.rerun()
            else:
                st.error("❌ Retraining failed. Please check strategy_1_retrain.py.")
        except Exception as e:
            st.error(f"❌ Error during retraining: {e}")

# === Manual Refresh ===
if st.button("🔃 Refresh Now"):
    st.rerun()

# === Strategy Lab Sliders ===
st.sidebar.markdown("### 🧪 Strategy Lab")
sma_window = st.sidebar.slider("SMA Window", min_value=5, max_value=30, value=10)
rsi_threshold = st.sidebar.slider("RSI Threshold", min_value=10, max_value=90, value=50)
st.sidebar.write(f"SMA Window: {sma_window}, RSI Threshold: {rsi_threshold}")
