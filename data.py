from growwapi import GrowwAPI
import pandas as pd
import datetime

# === Groww API Credentials ===
API_AUTH_TOKEN = "eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjE3NTIxOTM4MDAsImlhdCI6MTc1MjEyNDk3MCwibmJmIjoxNzUyMTI0OTcwLCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCI5YWIxYTAxZS1mZmI4LTQ5Y2YtYTMyOC00ZmRkZDYzMWVhYjRcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiMzNiNmI0ZmItYjUzYS00NGRmLWFhZGItMzA4N2RiMjU3NTczXCIsXCJkZXZpY2VJZFwiOlwiODU4ZDUyYmYtNjlmNC01ZjBmLTg0NDAtMGRmZTk0OGI3NjgxXCIsXCJzZXNzaW9uSWRcIjpcImVlODFmZjM4LWIwMjAtNDYwNy1hZjM0LWRjYTI0MmQ3Zjg5MlwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYk4zUFloTGFEZGt1ZlBCYU5zRDhLTnhSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcIm9yZGVyLWJhc2ljLGxpdmVfZGF0YS1iYXNpYyxub25fdHJhZGluZy1iYXNpYyxvcmRlcl9yZWFkX29ubHktYmFzaWNcIixcInNvdXJjZUlwQWRkcmVzc1wiOlwiNDkuMjA1LjI0Ni4xMjMsMTcyLjY4LjE2Ni4xNDUsMzUuMjQxLjIzLjEyM1wiLFwidHdvRmFFeHBpcnlUc1wiOjE3NTIxOTM4MDAwMDB9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.Q8AUUjRYNT1t7giofTCouO5HQbV0cTagosPJ7oAjMlV8WMAJwecDx3p1vZEDvxDRuK6oKHFjMDV1qbEQ_zYQPA"
groww = GrowwAPI(API_AUTH_TOKEN)

# === Load Local CSV ===
instruments_df = pd.read_csv("instruments.csv")

# ✅✅ FULL PATCH to prevent file writing in growwapi
groww.instruments = instruments_df
groww._load_instruments = lambda: None
groww._download_and_load_instruments = lambda: instruments_df

# === You can now safely call: groww.get_instrument_by_groww_symbol()
# === No permission errors will occur now

# Automated Date Setup
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

# last 30 days
start_time = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d 09:15:00")
end_time   = today.strftime("%Y-%m-%d 15:15:00")
new_live_data = groww.get_historical_candle_data("NIFTY", groww.EXCHANGE_NSE, groww.SEGMENT_CASH, start_time, end_time, 10)
