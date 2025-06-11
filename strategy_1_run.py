import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from strategy_1_model import buy_model, rr_model, compute_rsi
from data import groww

# --- SET START TIME (manually) ---
# Format: YYYY, MM, DD, HH, MM (24-hour clock)
start_time_ist = datetime(2025, 6, 10, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))

# --- SET END TIME (current time in IST) ---
end_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

# Debug
print("Start Time in IST:", start_time_ist)
print("End Time in IST:", end_time_ist)

# Convert to UTC if needed for API
start_time_utc = start_time_ist.astimezone(ZoneInfo("UTC"))
end_time_utc = end_time_ist.astimezone(ZoneInfo("UTC"))

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

# Call prediction
live_predict()

# Optional refresh button
if st.button("🔁 Refresh Now"):
    st.rerun()
