import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI

# === Page Settings ===
st.set_page_config(page_title="Trading Signal Predictor", layout="wide")
st.title("📊 Trading Signal Predictor")

# === Groww API Auth ===
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

groww = GrowwAPI(api_key)

# === Patch Instrument Loader ===
instruments_df = pd.read_csv("instruments.csv")
groww.instruments = instruments_df
groww._load_instruments = lambda: None
groww._download_and_load_instruments = lambda: instruments_df
groww.get_instrument_by_groww_symbol = lambda symbol: instruments_df[instruments_df['groww_symbol'] == symbol].iloc[0].to_dict()

# === Load Trained Models ===
try:
    buy_model = joblib.load("models/buy_model_latest.pkl")
    rr_model = joblib.load("models/rr_model_latest.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# === RSI Function ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === Time Setup ===
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# === Auto Refresh (Trading Hours Only) ===
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)

# === Main Prediction Function ===
def live_predict(symbol="NSE-NIFTY", interval_minutes=10):
    try:
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

        if not data or 'candles' not in data or not data['candles']:
            st.error("⚠️ No candle data returned from Groww API.")
            return

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
            st.warning("❌ Not enough data to generate prediction.")
            return

        # Predict
        buy_signal = buy_model.predict(latest)[0]
        rr_signal = rr_model.predict(latest)[0]

        # Output
        st.metric("🕒 Last Candle", df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"))
        st.metric("📈 Signal", "✅ BUY" if buy_signal == 1 else "❌ HOLD / SELL")
        st.metric("📊 Risk/Reward", f"{rr_signal:.4f}")
        st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")

# === Run Prediction ===
live_predict()

# === Manual Refresh ===
if st.button("🔁 Refresh Now"):
    st.rerun()
