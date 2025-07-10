import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from growwapi import GrowwAPI

st.set_page_config(page_title="📈 Trading Signal Predictor", layout="wide")
st.title("💹 Trading Signal Predictor")

# === Sidebar Auth ===
st.sidebar.header("🔐 Groww API Auth")
api_key = st.sidebar.text_input("Enter your Groww API token", type="password")

if not api_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# === Init Groww API ===
groww = GrowwAPI(api_key)
instruments_df = pd.read_csv("instruments.csv")
groww.instruments = instruments_df
groww._load_instruments = lambda: None
groww._download_and_load_instruments = lambda: instruments_df
groww.get_instrument_by_groww_symbol = lambda symbol: instruments_df[instruments_df['groww_symbol'] == symbol].iloc[0].to_dict()

# === Load Models ===
try:
    buy_model = joblib.load("models/buy_model_latest.pkl")
    rr_model = joblib.load("models/rr_model_latest.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# === Compute RSI ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === Trading Time Check ===
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
if datetime.strptime("09:15", "%H:%M").time() <= now_ist.time() <= datetime.strptime("15:30", "%H:%M").time():
    st.markdown("<meta http-equiv='refresh' content='600'>", unsafe_allow_html=True)

# === Symbol Selection ===
symbols = instruments_df['groww_symbol'].dropna().unique()
selected_symbol = st.sidebar.selectbox("📉 Select Symbol", sorted(symbols), index=list(symbols).index("NSE-NIFTY") if "NSE-NIFTY" in symbols else 0)

# === Tabs ===
tabs = st.tabs(["🔮 Live Prediction", "📅 Backtest"])

# === Common Parameters ===
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# === Prediction Logic ===
def predict_and_display(symbol, start_time, end_time, interval_minutes=10, backtest=False):
    try:
        selected = groww.get_instrument_by_groww_symbol(symbol)
        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=selected['exchange'],
            segment=selected['segment'],
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_minutes
        )

        if not data or 'candles' not in data or not data['candles']:
            st.error("⚠️ No candle data returned.")
            return

        df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['Volatility'] = df['close'].rolling(window=10).std()
        df['RSI'] = compute_rsi(df['close'])

        features = df[['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']].dropna()

        if features.empty:
            st.warning("Not enough data to make predictions.")
            return

        df = df.iloc[-len(features):]  # align
        df['Buy_Signal'] = buy_model.predict(features)
        df['Risk_Reward'] = rr_model.predict(features)

        if backtest:
            st.subheader("📅 Backtest Results")
            st.line_chart(df.set_index('timestamp')[['close', 'Risk_Reward']])
            st.dataframe(df[['timestamp', 'close', 'Buy_Signal', 'Risk_Reward']].tail(20), use_container_width=True)
        else:
            latest = df.iloc[-1]
            st.metric("🕒 Last Candle", latest['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
            st.metric("📈 Signal", "✅ BUY" if latest['Buy_Signal'] == 1 else "❌ HOLD / SELL")
            st.metric("📊 Risk/Reward", f"{latest['Risk_Reward']:.4f}")
            st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# === Tab: Live ===
with tabs[0]:
    predict_and_display(selected_symbol, start_time_ist, end_time_ist)

# === Tab: Backtest ===
with tabs[1]:
    backtest_start = st.date_input("📆 Backtest Start Date", value=datetime(2024, 6, 1).date())
    backtest_end = st.date_input("📆 Backtest End Date", value=datetime.now().date())

    if backtest_start >= backtest_end:
        st.warning("Start date must be before end date.")
    else:
        start_bt = datetime.combine(backtest_start, datetime.min.time()).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        end_bt = datetime.combine(backtest_end, datetime.max.time()).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        predict_and_display(selected_symbol, start_bt, end_bt, backtest=True)
