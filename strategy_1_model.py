import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
from data import load_data

# Read token from env or config
AUTH_TOKEN = os.getenv("GROWW_API_AUTH_TOKEN")
if not AUTH_TOKEN:
    raise ValueError("⚠️ GROWW_API_AUTH_TOKEN not set in environment.")

groww, df_4, df_3, _, _ = load_data(AUTH_TOKEN)

# --- Prepare DataFrame ---
def prepare_df(raw_data):
    df = pd.DataFrame(raw_data['candles'],
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

df_4 = prepare_df(df_4)
df_3 = prepare_df(df_3)

# Combine and clean
df = pd.concat([df_4, df_3], ignore_index=True)
df.drop_duplicates(subset=['timestamp'], inplace=True)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Feature Engineering ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower

def compute_stochastic(df):
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    k = ((df['close'] - low_14) / (high_14 - low_14)) * 100
    d = k.rolling(window=3).mean()
    return k, d

def compute_macd(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Indicators
df['SMA_10'] = df['close'].rolling(window=10).mean()
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)
df['Volatility'] = df['close'].rolling(window=10).std()
df['RSI'] = compute_rsi(df['close'])
bb_u, bb_l = compute_bollinger_bands(df['close'])
df['BB_Upper'] = bb_u
df['BB_Lower'] = bb_l
k, d = compute_stochastic(df)
df['Stoch_K'] = k
df['Stoch_D'] = d
macd, sig_line = compute_macd(df)
df['MACD'] = macd
df['Signal_Line'] = sig_line

# --- Labeling ---
df['Buy_Signal'] = 0
df.loc[(df['RSI'] < 30) | ((df['Momentum'] > 0) & (df['close'] > (2 * (df['SMA_10'] + df['EMA_10'])) / 3)), 'Buy_Signal'] = 1
df.loc[~((df['RSI'] > 70) | ((df['Momentum'] < 0) & (df['close'] < (2 * (df['SMA_10'] + df['EMA_10'])) / 3))), 'Buy_Signal'] = 1
df.loc[((k < 20) | (d < 20)) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
df.loc[((macd > sig_line) & (sig_line.isnull().all())) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1
df.loc[((df['close'] < df['BB_Lower']) | (df['close'] > df['BB_Upper'])) & (df['Buy_Signal'] == 0), 'Buy_Signal'] = 1

df['Risk_Reward'] = (df['high'] - df['low']) / df['close']

# --- Cleanup ---
req_cols = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 'Buy_Signal', 'Risk_Reward']
df.dropna(subset=req_cols, inplace=True)
df.ffill(inplace=True)

# --- Balance Dataset ---
df_1 = df[df['Buy_Signal'] == 1]
n_pos = len(df_1)
df_0 = df[df['Buy_Signal'] == 0].sample(n=n_pos, replace=(n_pos > len(df[df['Buy_Signal'] == 0])), random_state=42)
df_balanced = pd.concat([df_1, df_0]).sample(frac=1, random_state=42)

# --- Train Models ---
features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility']
X = df_balanced[features]
y_buy = df_balanced['Buy_Signal']
y_rr = df_balanced['Risk_Reward']

X_train, X_test, y_train_buy, y_test_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)
_, _, y_train_rr, y_test_rr = train_test_split(X, y_rr, test_size=0.2, random_state=42)

clf_params = {
    'n_estimators': [300, 500, 1000],
    'max_depth': [10, 30, 50, None],
    'min_samples_split': [2, 5, 10]
}
clf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    clf_params,
    n_iter=5,
    scoring='accuracy',
    cv=3,
    random_state=42
)
clf_search.fit(X_train, y_train_buy)
buy_model = clf_search.best_estimator_

rr_model = RandomForestRegressor(n_estimators=3000, max_depth=50, random_state=42)
rr_model.fit(X_train, y_train_rr)

y_pred_buy = buy_model.predict(X_test)
y_pred_rr = rr_model.predict(X_test)

print("✅ Buy Model Accuracy:", accuracy_score(y_test_buy, y_pred_buy))
print("✅ Risk-Reward MSE:", mean_squared_error(y_test_rr, y_pred_rr))

# --- Save Models ---
os.makedirs("models", exist_ok=True)
joblib.dump(buy_model, 'models/buy_model_latest.pkl')
joblib.dump(rr_model, 'models/rr_model_latest.pkl')
print("💾 Models saved in 'models/' directory.")

# --- Export reusable RSI function ---
def compute_rsi_for_live(series, period=14):
    return compute_rsi(series, period)
compute_rsi = compute_rsi_for_live
