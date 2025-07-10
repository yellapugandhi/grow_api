import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI
import numpy as np
from functools import lru_cache
import subprocess
import sys

# 1. Page configuration and header
st.set_page_config(page_title="Trading Signal Predictor", layout="wide")
st.title("📈 Trading Signal Predictor")
st.write("🔄 Initializing…")

# 2. Groww API Auth
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")
st.write("🔑 API Key registered:", bool(api_key))

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

groww = GrowwAPI(api_key)

# 3. Cached Instrument Metadata
@lru_cache(maxsize=1)
def load_instruments():
    try:
        df = pd.read_csv("instruments.csv", low_memory=False)
        st.write("✅ instruments.csv loaded:", df.shape)
    except Exception as e:
        st.error(f"Failed to load instruments.csv: {e}")
        raise

    # Override GrowwAPI methods to use local CSV
    groww.instruments = df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: df
    groww.get_instrument_by_groww_symbol = lambda symbol: (
        df[df['groww_symbol'] == symbol].iloc[0].to_dict()
    )
    return df

try:
    instruments_df = load_instruments()
except:
    st.stop()

# 4. Load Models
try:
    from strategy_1_model import buy_model, rr_model, compute_rsi
    st.write("✅ Models loaded successfully")
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# 5. Set Date Range
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
now_ist = end_time_ist

# 6. Auto Refresh (during trading hours)
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown(
        "<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True
    )

# 7. Strategy Dropdown
strategy_option = st.sidebar.selectbox("Select Strategy Version", ["Strategy 1"])

# 8. Live Prediction Function
def live_predict(symbol="NSE-NIFTY", interval_minutes=10):
    st.subheader("📡 Fetching live data…")
    st.write("• Symbol:", symbol)
    st.write("• Interval (minutes):", interval_minutes)

    # Build time strings
    start_str = start_time_ist.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time_ist.strftime("%Y-%m-%d %H:%M:%S")

    # Instrument lookup
    try:
        selected = groww.get_instrument_by_groww_symbol(symbol)
        st.write("🧾 Instrument selected:", selected)
    except Exception as e:
        st.error(f"Instrument lookup failed: {e}")
        return

    # API call
    try:
        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=selected['exchange'],
            segment=selected['segment'],
            start_time=start_str,
            end_time=end_str,
            interval_in_minutes=interval_minutes
        )
        st.write("📡 Raw Groww response:", data)
    except Exception as e:
        st.error(f"Groww API call failed: {e}")
        return

    # Validate response
    candles = data.get('candles') if isinstance(data, dict) else None
    count = len(candles) if candles else 0
    st.write("📈 Candle count:", count)

    if not candles:
        st.error("⚠️ No candle data returned from Groww API.")
        return

    # Build DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = (
        pd.to_datetime(df['timestamp'], unit='s', utc=True)
        .dt.tz_convert('Asia/Kolkata')
    )
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

    # Model inference
    try:
        proba = buy_model.predict_proba(latest)[0]
        rr_signal = rr_model.predict(latest)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    confidence = proba[1] * 100
    buy_signal = int(proba[1] > 0.5)

    # Display results
    st.subheader("📊 Prediction Results")
    st.write("• Last Candle Time:", df['timestamp'].iloc[-1])
    st.write("• Signal:", "BUY" if buy_signal else "HOLD / SELL")
    st.write(f"• Confidence: {confidence:.2f}%")
    st.write(f"• Risk/Reward: {rr_signal:.4f}")
    st.dataframe(df.tail(10), use_container_width=True)

    # Candle Timer
    next_candle = df['timestamp'].iloc[-1] + timedelta(minutes=interval_minutes)
    remaining = next_candle - datetime.now(ZoneInfo("Asia/Kolkata"))
    st.info(f"⏳ Time until next candle: {remaining.seconds // 60}m {remaining.seconds % 60}s")

# 9. Run live prediction
live_predict()

# 10. Retrain Model Trigger
st.sidebar.markdown("### 🧠 Retrain")
if st.sidebar.button("🔁 Retrain Model"):
    st.info("📡 Starting retraining…")
    try:
        process = subprocess.Popen(
            [sys.executable, "strategy_1_retrain.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        output_area = st.empty()
        logs = ""
        for line in process.stdout:
            logs += line
            output_area.code(logs)
        process.wait()

        if process.returncode == 0:
            st.success("✅ Retraining complete!")
            st.experimental_rerun()
        else:
            st.error("❌ Retraining failed. Check `strategy_1_retrain.py` logs.")
    except Exception as e:
        st.error(f"❌ Retraining error: {e}")

# 11. Manual Refresh
if st.button("🔃 Refresh Now"):
    st.experimental_rerun()
