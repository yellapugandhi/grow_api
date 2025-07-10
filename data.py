import pandas as pd
import datetime

def load_data(groww):
    instruments_df = pd.read_csv("instruments.csv")

    # Patch to avoid permission errors
    groww.instruments = instruments_df
    groww._load_instruments = lambda: None
    groww._download_and_load_instruments = lambda: instruments_df

    today = datetime.datetime.now()

    # 4th month ago
    start_time = (today - datetime.timedelta(days=120)).strftime("%Y-%m-%d 09:15:00")
    end_time   = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d 15:15:00")
    df_4 = groww.get_historical_candle_data("NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start_time, end_time, 240)

    # 3rd month ago
    start_time = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d 09:15:00")
    end_time   = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d 15:15:00")
    df_3 = groww.get_historical_candle_data("NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start_time, end_time, 60)

    # 2nd month ago
    start_time = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d 09:15:00")
    end_time   = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d 15:15:00")
    df_2 = groww.get_historical_candle_data("NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start_time, end_time, 10)

    # Last 30 days
    start_time = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d 09:15:00")
    end_time   = today.strftime("%Y-%m-%d 15:15:00")
    new_live_data = groww.get_historical_candle_data("NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start_time, end_time, 10)

    return df_4, df_3, df_2, new_live_data
