from growwapi import GrowwAPI
import pandas as pd
import datetime

# === Groww API Credentials ===
API_AUTH_TOKEN = "eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjE3NTIxOTM4MDAsImlhdCI6MTc1MjEyNDk3MCwibmJmIjoxNzUyMTI0OTcwLCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCI5YWIxYTAxZS1mZmI4LTQ5Y2YtYTMyOC00ZmRkZDYzMWVhYjRcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiMzNiNmI0ZmItYjUzYS00NGRmLWFhZGItMzA4N2RiMjU3NTczXCIsXCJkZXZpY2VJZFwiOlwiODU4ZDUyYmYtNjlmNC01ZjBmLTg0NDAtMGRmZTk0OGI3NjgxXCIsXCJzZXNzaW9uSWRcIjpcImVlODFmZjM4LWIwMjAtNDYwNy1hZjM0LWRjYTI0MmQ3Zjg5MlwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYk4zUFloTGFEZGt1ZlBCYU5zRDhLTnhSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcIm9yZGVyLWJhc2ljLGxpdmVfZGF0YS1iYXNpYyxub25fdHJhZGluZy1iYXNpYyxvcmRlcl9yZWFkX29ubHktYmFzaWNcIixcInNvdXJjZUlwQWRkcmVzc1wiOlwiNDkuMjA1LjI0Ni4xMjMsMTcyLjY4LjE2Ni4xNDUsMzUuMjQxLjIzLjEyM1wiLFwidHdvRmFFeHBpcnlUc1wiOjE3NTIxOTM4MDAwMDB9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.Q8AUUjRYNT1t7giofTCouO5HQbV0cTagosPJ7oAjMlV8WMAJwecDx3p1vZEDvxDRuK6oKHFjMDV1qbEQ_zYQPA"
groww = GrowwAPI(API_AUTH_TOKEN)

# === Get All Instruments ===
instruments_df = groww.get_all_instruments()
print(instruments_df.head())

instruments_df['expiry_date'] = pd.to_datetime(instruments_df['expiry_date'])

instruments_df[
    (instruments_df['underlying_symbol'] == 'NIFTY') &
    (instruments_df['strike_price'] == '25100') &
    (instruments_df['instrument_type'] == 'CE') &
    (instruments_df['expiry_date'] == '2025-06-12')
]

# === Automated Date Setup ===
today = datetime.datetime.now()

# 4th month ago
start_time = (today - datetime.timedelta(days=120)).strftime("%Y-%m-%d 09:15:00")
end_time = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d 15:15:00")

historical_data_120days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=240
)
print(historical_data_120days)
df_4 = historical_data_120days

# 3rd month ago
start_time = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d 09:15:00")
end_time = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d 15:15:00")

historical_data_90days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=60
)
print(historical_data_90days)
df_3 = historical_data_90days

# 2nd month ago
start_time = (today - datetime.timedelta(days=60)).strftime("%Y-%m-%d 09:15:00")
end_time = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d 15:15:00")

historical_data_60days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=10
)
print(historical_data_60days)
df_2 = historical_data_60days

# last 30 days
start_time = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d 09:15:00")
end_time = today.strftime("%Y-%m-%d 15:15:00")

last_data_30days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=10
)
print(last_data_30days)
new_live_data = last_data_30days
