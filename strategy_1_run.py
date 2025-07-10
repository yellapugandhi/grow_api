import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from time import sleep

# === Groww API Auth ===
from growwapi import GrowwAPI

st.set_page_config(page_title="Trading Signal Predictor", layout="wide")
st.sidebar.title("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

groww = GrowwAPI(api_key)

# Load instruments.csv and patch GrowwAPI to avoid permission errors
instruments_df = pd.read_csv("instruments.csv")
groww.instruments = instruments_df
groww._load_instruments = lambda: None
groww._download_and_load_instruments = lambda: instruments_df
groww.get_instrument_by_groww_symbol = lambda symbol: instruments_df[instruments_df['groww_symbol'] == symbol].iloc[0].to_dict()

# === Import Models ===
from strategy_1_model import buy_model, rr_model, compute_rsi

# --- SET TIME RANGE ---
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# Auto-refresh logic (if within trading hours)
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
if now_ist.time() >= datetime.strptime("09:15", "%H:%M").time() and now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st_autorefresh = st.experimental_rerun
    st.experimental_set_query_params(run=str(now_ist))  # Force query param change
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)  # Refresh every 600 sec = 10 min

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

        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['Volatility'] = df['close'].rolling(window=10).std()
        df['RSI'] = compute_rsi(df['close'])

        latest = df[['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']].dropna().tail(1)

        if latest.empty:
            st.warning("Not enough data to predict.")
            return

        buy_signal = buy_model.predict(latest)[0]
        rr_signal = rr_model.predict(latest)[0]

        st.metric("🕒 Last Candle", df['timestamp'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"))
        st.metric("📈 Signal", "BUY" if buy_signal == 1 else "HOLD / SELL")
        st.metric("📊 Risk/Reward", f"{rr_signal:.4f}")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.error("⚠️ No candle data returned from Groww API.")

# === Call Prediction ===
live_predict()

# === Optional Manual Refresh Button ===
if st.button("🔁 Refresh Now"):
    st.rerun()
