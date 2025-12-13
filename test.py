# hybrid_lgbm_rnn_pipeline.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ----------------------------
# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ----------------------------
# CONFIG
WINDOW = 48               # number of past timesteps for RNN
FUTURE_PERIODS = 6        # target = sum of next 6 funding rates
TEST_RATIO = 0.1
VAL_RATIO = 0.1           # fraction of train to hold-out for stacking
BATCH_SIZE = 64
EPOCHS = 50

# ----------------------------
# 1) LOAD DATA
# Expect a DataFrame `df` with:
# timestamp (datetime), columns for spot: spot_open, spot_high, spot_low, spot_close, spot_volume, ...
# and future: fut_open, fut_close, ..., and funding_rate column (per period)
# Adjust column names if different.
#
# Example: df = pd.read_csv("merged_spot_future.csv", parse_dates=["timestamp"])
# For this pipeline I'll assume df already merged with prefix 'spot_' and 'fut_',
# and funding rate column 'funding_rate' (which is the rate at that timestamp)
# ----------------------------
funding = pd.read_csv("D:\\Homework\\QF634\\project\\TAOUSDT_funding_rate_20200101_20251130.csv", parse_dates=["fundingDateTime"], date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S.%f"))
funding = funding.sort_values("fundingDateTime").drop("formattedFundingDateTime",axis = 1).drop("symbol",axis = 1)
funding.rename(columns={'fundingDateTime': 'timestamp','fundingRate':"funding_rate"}, inplace=True)
spot = pd.read_json('D:\\Homework\\QF634\\project\\Data\\raw_historical_price\\TAOUSDT_5m_binance_spot_historical_data.json',lines=True)           # columns: timestamp, open, high, low, close, volume, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
future = pd.read_json('D:\\Homework\\QF634\\project\\Data\\raw_historical_price\\TAOUSDT_5m_binance_futures_historical_data.json',lines=True) 
funding.set_index('timestamp', inplace=True)
spot.set_index('timestamp', inplace=True)
future.set_index('timestamp', inplace=True)
df_future_renamed = future.add_prefix("fut_")
df_spot_renamed = spot.add_prefix("spot_")
# Important: rename the key column back so merge works
df_future_renamed = df_future_renamed.rename(columns={"fut_timestamp": "timestamp"})
df_spot_renamed = df_spot_renamed.rename(columns={"spot_timestamp": "timestamp"})

df = funding.merge(df_spot_renamed, left_index=True, right_index=True, how='left').merge(df_future_renamed, left_index=True, right_index=True, how='left', )

# quick sanity
print("Rows:", len(df), "Columns:", df.shape[1])
print(df.columns)
# ----------------------------
# 2) FEATURE ENGINEERING (per-timestep)
# ----------------------------
# basic helper
def pct_change(series, periods=1):
    return series.pct_change(periods)

# Spot features
df["spot_ret_1"] = pct_change(df["spot_close"], 1)
df["spot_ret_6"] = pct_change(df["spot_close"], 6)
df["spot_vol_24"] = df["spot_close"].pct_change().rolling(24).std()
df["spot_vol_48"] = df["spot_close"].pct_change().rolling(48).std()
df["spot_rsi_14"] = (df["spot_close"].diff(1).clip(lower=0).rolling(14).mean() /
                     df["spot_close"].diff(1).abs().rolling(14).mean()).fillna(0)

# Future features
df["fut_ret_1"] = pct_change(df["fut_close"], 1)
df["fut_ret_6"] = pct_change(df["fut_close"], 6)
df["fut_vol_24"] = df["fut_close"].pct_change().rolling(24).std()
df["fut_vol_48"] = df["fut_close"].pct_change().rolling(48).std()

# Spread and funding stats
df["spread_close"] = df["fut_close"] - df["spot_close"]
df["spread_ret_1"] = df["spread_close"].pct_change()
df["funding_mean_6"] = df["funding_rate"].rolling(6).mean()
df["funding_std_6"]  = df["funding_rate"].rolling(6).std()
df["funding_sum_6_past"] = df["funding_rate"].rolling(6).sum()

# Liquidity / order-flow derived features (ratios)
df["taker_buy_ratio_spot"] = df["spot_taker_buy_base_asset_volume"] / (df["spot_volume"].replace(0, np.nan))
df["taker_buy_ratio_fut"]  = df["fut_taker_buy_base_asset_volume"] / (df["fut_volume"].replace(0, np.nan))

# Fill / clip
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ----------------------------
# 3) TARGET: sum of next FUTURE_PERIODS funding_rate values
# For each timestamp t we predict sum of funding_rate at t+1 .. t+FUTURE_PERIODS
# ----------------------------
df["y"] = df["funding_rate"].shift(-1).rolling(FUTURE_PERIODS).sum().shift(-(FUTURE_PERIODS-1))
# Another clearer way (same result):
# df['y'] = df['funding_rate'].shift(-1).fillna(0)
# for i in range(2, FUTURE_PERIODS+1):
#     df['y'] += df['funding_rate'].shift(-i)

# Drop the last rows that don't have full future target
df = df.iloc[:-FUTURE_PERIODS].reset_index(drop=True)

# ----------------------------
# 4) Build sequence samples aligned with tabular sample (last row index of each sequence)
# Each sequence X_sample = features[t - WINDOW + 1 : t + 1]
# The label y_sample = sum funding_rate at t+1 .. t+FUTURE_PERIODS
# We'll prepare:
# - seq_features (3D array) for RNN
# - tab_features (2D array) based on last row in sequence for LGBM
# - y
# ----------------------------
feature_cols = [
    # choose a reasonable subset of engineered + raw features
    "spot_close", "spot_ret_1", "spot_ret_6", "spot_vol_24", "spot_vol_48", "spot_rsi_14",
    "fut_close", "fut_ret_1", "fut_ret_6", "fut_vol_24", "fut_vol_48",
    "spread_close", "spread_ret_1",
    "funding_mean_6", "funding_std_6", "funding_sum_6_past",
    "spot_volume", "fut_volume",
    "taker_buy_ratio_spot", "taker_buy_ratio_fut"
]

# Ensure all features are in df
feature_cols = [c for c in feature_cols if c in df.columns]
print("Using feature columns:", feature_cols)

seq_X = []
tab_X = []
y = []
last_idx_list = []

for start in range(0, len(df) - WINDOW - FUTURE_PERIODS + 1):
    end = start + WINDOW  # end is exclusive; last index included in sequence = end-1
    seq_slice = df.loc[start:end-1, feature_cols].values  # shape (WINDOW, n_features)
    last_idx = end - 1
    # target is sum of funding_rate at indices end .. end+FUTURE_PERIODS-1
    y_val = df.loc[end:end + FUTURE_PERIODS - 1, "funding_rate"].sum()
    seq_X.append(seq_slice)
    tab_X.append(df.loc[last_idx, feature_cols].values)  # tabular snapshot at last timestep
    y.append(y_val)
    last_idx_list.append(last_idx)

seq_X = np.array(seq_X)            # (n_samples, WINDOW, n_features)
tab_X = np.array(tab_X)            # (n_samples, n_features)
y = np.array(y)                    # (n_samples,)
last_idx_list = np.array(last_idx_list)

print("Samples:", seq_X.shape[0], "Sequence shape:", seq_X.shape[1:], "Tab shape:", tab_X.shape[1:])

# ----------------------------
# 5) Chronological split: train / val / test on samples
# We'll first split test out, then within train split validation for stacking
# ----------------------------
n_samples = seq_X.shape[0]
test_size = int(n_samples * TEST_RATIO)
val_size = int(n_samples * VAL_RATIO)

train_end = n_samples - test_size
val_start = int(train_end * (1 - VAL_RATIO))  # small chronological holdout within train

# Simpler: split as train / val / test chronologically:
train_idx = np.arange(0, int(n_samples * 0.8))
val_idx   = np.arange(int(n_samples * 0.8), int(n_samples * 0.9))
test_idx  = np.arange(int(n_samples * 0.9), n_samples)

X_seq_train, X_seq_val, X_seq_test = seq_X[train_idx], seq_X[val_idx], seq_X[test_idx]
X_tab_train, X_tab_val, X_tab_test = tab_X[train_idx], tab_X[val_idx], tab_X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print("Train / Val / Test sizes:", len(train_idx), len(val_idx), len(test_idx))

# ----------------------------
# 6) Scale features
# - For LGBM: scaling is optional, but we'll scale tabular features for stability
# - For RNN: scale input sequences with a StandardScaler fitted on flattened train sequences
# - Scale target y for RNN training; LGBM trains on raw target (but we could also use scaled)
# ----------------------------
# Tabular scaler
tab_scaler = StandardScaler()
X_tab_train_scaled = tab_scaler.fit_transform(X_tab_train)
X_tab_val_scaled   = tab_scaler.transform(X_tab_val)
X_tab_test_scaled  = tab_scaler.transform(X_tab_test)

# Sequence scaler: fit on concatenation of all timesteps in train sequences
n_feat = X_seq_train.shape[2]
seq_flat = X_seq_train.reshape(-1, n_feat)
seq_scaler = StandardScaler().fit(seq_flat)

def scale_sequences(seqs, scaler):
    s = seqs.copy()
    n_samples = s.shape[0]
    s = s.reshape(-1, s.shape[2])
    s = scaler.transform(s)
    return s.reshape(n_samples, WINDOW, s.shape[1])

X_seq_train_scaled = scale_sequences(X_seq_train, seq_scaler)
X_seq_val_scaled   = scale_sequences(X_seq_val, seq_scaler)
X_seq_test_scaled  = scale_sequences(X_seq_test, seq_scaler)

# Target scaler for RNN (helpful when target small)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

# ----------------------------
# 7) Train LightGBM on tabular features (train -> val)
# ----------------------------
lgb_train = lgb.Dataset(X_tab_train_scaled, label=y_train)
lgb_val = lgb.Dataset(X_tab_val_scaled, label=y_val, reference=lgb_train)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": SEED,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
)

# LGBM predictions
pred_lgb_val = lgb_model.predict(X_tab_val_scaled, num_iteration=lgb_model.best_iteration)
pred_lgb_test = lgb_model.predict(X_tab_test_scaled, num_iteration=lgb_model.best_iteration)

# ----------------------------
# 8) Build & train RNN (LSTM)
# ----------------------------
tf.keras.backend.clear_session()

rnn_model = Sequential([
    LSTM(128, input_shape=(WINDOW, n_feat), return_sequences=False),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

rnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss", verbose=1)
]

history = rnn_model.fit(
    X_seq_train_scaled, y_train_scaled,
    validation_data=(X_seq_val_scaled, y_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# RNN predictions (remember to inverse transform)
pred_rnn_val_scaled = rnn_model.predict(X_seq_val_scaled).flatten()
pred_rnn_test_scaled = rnn_model.predict(X_seq_test_scaled).flatten()

pred_rnn_val = y_scaler.inverse_transform(pred_rnn_val_scaled.reshape(-1,1)).flatten()
pred_rnn_test = y_scaler.inverse_transform(pred_rnn_test_scaled.reshape(-1,1)).flatten()

# ----------------------------
# 9) Stacking / Meta model
# Use val predictions from both models to train a simple linear regressor as meta model
# ----------------------------
stack_train = np.vstack([pred_lgb_val, pred_rnn_val]).T
meta = LinearRegression().fit(stack_train, y_val)

# Make stacked predictions on test set
stack_test = np.vstack([pred_lgb_test, pred_rnn_test]).T
pred_stack_test = meta.predict(stack_test)

# Option: simple average
pred_avg_test = (pred_lgb_test + pred_rnn_test) / 2.0

# ----------------------------
# 10) Evaluation
# ----------------------------
def print_metrics(true, pred, tag=""):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    print(f"{tag} RMSE: {rmse:.6f}  MAE: {mae:.6f}")

print_metrics(y_test, pred_lgb_test, "LGBM test")
print_metrics(y_test, pred_rnn_test, "RNN test")
print_metrics(y_test, pred_avg_test, "Average test")
print_metrics(y_test, pred_stack_test, "Stacked (meta) test")

# ----------------------------
# 11) Optional: Save models & scalers
# ----------------------------
lgb_model.save_model("lgb_model.txt")
rnn_model.save("rnn_model.h5")
import joblib
joblib.dump(tab_scaler, "tab_scaler.pkl")
joblib.dump(seq_scaler, "seq_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")
joblib.dump(meta, "meta_linear.pkl")

print("Models saved: lgb_model.txt, rnn_model.h5, scalers + meta.")

# ----------------------------
# 12) Quick visualization of predictions vs actual (optional)
# ----------------------------
try:
    import matplotlib.pyplot as plt
    idx = np.arange(len(y_test))
    plt.figure(figsize=(12,4))
    plt.plot(idx, y_test, label="actual")
    plt.plot(idx, pred_lgb_test, label="lgbm")
    plt.plot(idx, pred_rnn_test, label="rnn")
    plt.plot(idx, pred_stack_test, label="stacked")
    plt.legend()
    plt.title("Test predictions vs actual (sample-level)")
    plt.show()
except Exception as e:
    print("matplotlib not available or plotting failed:", e)