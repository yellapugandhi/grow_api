from growwapi import GrowwAPI
import pandas as pd
import datetime
 
# Groww API Credentials (Replace with your actual credentials)
API_AUTH_TOKEN = "eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjE3NTAxMjAyMDAsImlhdCI6MTc1MDA1MjQyNCwibmJmIjoxNzUwMDUyNDI0LCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCIyODA0NjY0Yi1lYTQxLTQ3YjItOTg1OS0wYTU4MDljZTBiNmZcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiMzNiNmI0ZmItYjUzYS00NGRmLWFhZGItMzA4N2RiMjU3NTczXCIsXCJkZXZpY2VJZFwiOlwiODU4ZDUyYmYtNjlmNC01ZjBmLTg0NDAtMGRmZTk0OGI3NjgxXCIsXCJzZXNzaW9uSWRcIjpcImQ5YmFlODZhLTIyZTctNDMyNy1iYWQ3LWJmMDQ1OTIyOTRhN1wiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYk4zUFloTGFEZGt1ZlBCYU5zRDhLTnhYeVpHZmF5OEk2cVpSdER6Q3ZIQ0tcIixcInJvbGVcIjpcIm9yZGVyLWJhc2ljLGxpdmVfZGF0YS1iYXNpYyxub25fdHJhZGluZy1iYXNpYyxvcmRlcl9yZWFkX29ubHktYmFzaWNcIixcInZhbGlkSXBzXCI6XCJcIixcInNvdXJjZUlwQWRkcmVzc1wiOlwiNDkuMjA1LjI0Ni4xMjMsMTcyLjY4LjE2Ni4xNTksMzUuMjQxLjIzLjEyM1wiLFwidHdvRmFFeHBpcnlUc1wiOjE3NTAxMjAyMDAwMDB9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.qRSTxa3tONUFEYZSx-5dcyIFWfysUkCYwxEsiBSUjYk_HQi9tTzEGuHC3aSfr-n0t-TCvffIVhmJS8cFXEcSBA"
# Initialize Groww API
groww = GrowwAPI(API_AUTH_TOKEN)

instruments_df = groww.get_all_instruments()
print(instruments_df.head())


instruments_df['expiry_date'] = pd.to_datetime(instruments_df['expiry_date'])

instruments_df[
    (instruments_df['underlying_symbol'] == 'NIFTY') &
    (instruments_df['strike_price'] == '25100') &
    (instruments_df['instrument_type'] == 'CE') &
    (instruments_df['expiry_date'] == '2025-06-12')
    ]

# last 4th month data
start_time = "2025-02-10 09:15:00"
end_time = "2025-03-09 15:15:00"
 
historical_data_120days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=240 # Increased interval to 60 minutes
)
print(historical_data_120days)
df_4 = historical_data_120days

# last 3rd month data
start_time = "2025-03-10 09:15:00"
end_time = "2025-04-09 15:15:00"
 
historical_data_90days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes= 60 # Changed interval to 1 day
)
print(historical_data_90days)
df_3 = historical_data_90days

# last 2nd month data
start_time = "2025-04-10 09:15:00"
end_time = "2025-05-09 15:15:00"
 
historical_data_60days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=10 # Changed interval to 10 minutes
)
print(historical_data_60days)

df_2 = historical_data_60days

# last_data_30days
start_time = "2025-05-11 09:15:00"
end_time = "2025-06-10 15:15:00"
 
last_data_30days = groww.get_historical_candle_data(
    trading_symbol="NIFTY",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time=start_time,
    end_time=end_time,
    interval_in_minutes=10 # Changed interval to 10 minutes
)
print(last_data_30days)

new_live_data = last_data_30days