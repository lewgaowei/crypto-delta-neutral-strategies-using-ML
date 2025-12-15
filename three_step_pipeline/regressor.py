import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

funding = pd.read_csv(
    r"D:\Homework\QF634\project\TAOUSDT_funding_rate_20200101_20251130.csv",
    parse_dates=["fundingDateTime"],
    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S.%f"),
)
funding = funding.sort_values("fundingDateTime")
funding = funding.rename(columns={"fundingDateTime": "timestamp", "fundingRate": "funding_rate"})
funding = funding.drop(columns=["symbol", "formattedFundingDateTime"])
funding.set_index("timestamp", inplace=True)

spot = pd.read_json(
    r"D:\Homework\QF634\project\Data\raw_historical_price\TAOUSDT_5m_binance_spot_historical_data.json",
    lines=True,
).sort_values("timestamp").set_index("timestamp")

future = pd.read_json(
    r"D:\Homework\QF634\project\Data\raw_historical_price\TAOUSDT_5m_binance_futures_historical_data.json",
    lines=True,
).sort_values("timestamp").set_index("timestamp")

spot.index = pd.to_datetime(spot.index)
future.index = pd.to_datetime(future.index)
funding.index = pd.to_datetime(funding.index)

#Remove data from 15 mins before funding rate

fund_times = funding.index.sort_values()

clean_spot_windows = {}
clean_future_windows = {}

for i in range(1, len(fund_times)):
    t_prev = fund_times[i - 1]
    t_curr = fund_times[i]

    window_start = t_prev
    window_end = t_curr - pd.Timedelta(minutes=30)  # remove final 15m

    spot_window = spot.loc[(spot.index >= window_start) & (spot.index < window_end)]
    fut_window = future.loc[(future.index >= window_start) & (future.index < window_end)]

    clean_spot_windows[t_curr] = spot_window
    clean_future_windows[t_curr] = fut_window

# ==========================================================
# 4. MANUAL RESAMPLING (OHLCV)
# ==========================================================
spot_rows = []
future_rows = []

for t in fund_times[1:]:  # skip first because no full window
    sw = clean_spot_windows[t]
    fw = clean_future_windows[t]

    # ---- Spot ----
    spot_rows.append(pd.Series({
        "spot_open": sw["open"].iloc[0] if len(sw) else np.nan,
        "spot_high": sw["high"].max() if len(sw) else np.nan,
        "spot_low": sw["low"].min() if len(sw) else np.nan,
        "spot_close": sw["close"].iloc[-1] if len(sw) else np.nan,
        "spot_volume": sw["volume"].sum() if len(sw) else 0.0,
        "spot_quote_volume": sw["quote_asset_volume"].sum() if len(sw) else 0.0,
        "spot_trades": sw["number_of_trades"].sum() if len(sw) else 0.0,
        "spot_taker_buy_base": sw["taker_buy_base_asset_volume"].sum() if len(sw) else 0.0,
        "spot_taker_buy_quote": sw["taker_buy_quote_asset_volume"].sum() if len(sw) else 0.0,
    }, name=t))

    # ---- Future ----
    future_rows.append(pd.Series({
        "fut_open": fw["open"].iloc[0] if len(fw) else np.nan,
        "fut_high": fw["high"].max() if len(fw) else np.nan,
        "fut_low": fw["low"].min() if len(fw) else np.nan,
        "fut_close": fw["close"].iloc[-1] if len(fw) else np.nan,
        "fut_volume": fw["volume"].sum() if len(fw) else 0.0,
        "fut_quote_volume": fw["quote_asset_volume"].sum() if len(fw) else 0.0,
        "fut_trades": fw["number_of_trades"].sum() if len(fw) else 0.0,
        "fut_taker_buy_base": fw["taker_buy_base_asset_volume"].sum() if len(fw) else 0.0,
        "fut_taker_buy_quote": fw["taker_buy_quote_asset_volume"].sum() if len(fw) else 0.0,
    }, name=t))

spot_8h = pd.DataFrame(spot_rows)
future_8h = pd.DataFrame(future_rows)

# ==========================================================
# 5. MERGE ALL 4H DATA
# ==========================================================
df = funding.merge(spot_8h, left_index=True, right_index=True, how="left") \
            .merge(future_8h, left_index=True, right_index=True, how="left")

# ----------------------------------------------------------
# ASSUME df has: 
# - funding_rate column
# - is_spike column (from classifier labeling)
# - all engineered features used for classifier
# ----------------------------------------------------------

FUNDING_COL = "funding_rate"   # change as needed

df = df.sort_index().copy()
df = df.dropna(subset=[FUNDING_COL])

# ======================================================
# 1. FUNDING LAG FEATURES
# ======================================================
lags = [1, 2, 3, 6, 12]   # 1 lag = 4 hours, 6 = 1 day, 12 = 2 days

for L in lags:
    df[f"funding_lag_{L}"] = df[FUNDING_COL].shift(L)

# ======================================================
# 2. FUNDING ROLLING WINDOWS (4h * N)
# ======================================================
df["funding_vol_3"] = df[FUNDING_COL].rolling(3).std()
df["funding_vol_6"] = df[FUNDING_COL].rolling(6).std()
df["funding_vol_12"] = df[FUNDING_COL].rolling(12).std()

df["funding_momentum_3"] = df[FUNDING_COL] - df[FUNDING_COL].shift(3)
df["funding_momentum_6"] = df[FUNDING_COL] - df[FUNDING_COL].shift(6)

# Rolling max drawdown over N periods
def rolling_max_drawdown(series, window):
    roll = series.rolling(window)
    return (roll.max() - series) / roll.max()

df["funding_mdd_6"] = rolling_max_drawdown(df[FUNDING_COL], 6)
df["funding_mdd_12"] = rolling_max_drawdown(df[FUNDING_COL], 12)

# ======================================================
# 3. BASIS FEATURES (future - spot)
# ======================================================
df["basis"] = df["fut_close"] - df["spot_close"]
df["basis_change"] = df["basis"] - df["basis"].shift(1)
df["basis_lag_1"] = df["basis"].shift(1)
df["basis_lag_3"] = df["basis"].shift(3)

# ======================================================
# 4. PRICE MOMENTUM (spot + future)
# ======================================================
for c in ["spot_close", "fut_close"]:
    df[f"{c}_ret_1"] = df[c].pct_change(1)
    df[f"{c}_ret_3"] = df[c].pct_change(3)
    df[f"{c}_ret_6"] = df[c].pct_change(6)

# ======================================================
# 5. PRICE VOLATILITY (spot + future)
# ======================================================
for c in ["spot_close", "fut_close"]:
    df[f"{c}_vol_3"] = df[c].pct_change().rolling(3).std()
    df[f"{c}_vol_6"] = df[c].pct_change().rolling(6).std()

# ======================================================
# 6. VOLUME PRESSURE FEATURES
# ======================================================
df["spot_buy_ratio"] = df["spot_taker_buy_base"] / (df["spot_volume"] + 1e-9)
df["fut_buy_ratio"]  = df["fut_taker_buy_base"] / (df["fut_volume"] + 1e-9)

df["spot_buy_ratio_lag1"] = df["spot_buy_ratio"].shift(1)
df["fut_buy_ratio_lag1"]  = df["fut_buy_ratio"].shift(1)

# ======================================================
# REMOVE ROWS WITH NAN FROM LAGS
# ======================================================
df = df.dropna()

# ======================================================
# 7. SPIKE LABEL CREATION
# ======================================================
SPIKE_Q = 0.90
spike_threshold = df[FUNDING_COL].abs().quantile(SPIKE_Q)
df["is_spike"] = (df[FUNDING_COL].abs() >= spike_threshold).astype(int)
drop_cols = [FUNDING_COL, "is_spike"]
feature_cols = [c for c in df.columns if c not in drop_cols]

#Model Prediction Step 1:

df_spike = df[df["is_spike"] == 1].copy()
df_spike["spike_mag"] = df_spike[FUNDING_COL].abs()

y = df_spike["spike_mag"]
X = df_spike[feature_cols].dropna()
y = y.loc[X.index]

from catboost import CatBoostRegressor, Pool

threshold = y.quantile(0.75)   # top 25% = large spikes

small_mask = y <= threshold
large_mask = y > threshold

# =============================================================
# 3. Time-based split
# =============================================================
split = int(len(X) * 0.8)

# 1. Perform time split on full dataset
X_train_all = X.iloc[:split]
X_test_all  = X.iloc[split:]

y_train_all = y.iloc[:split]
y_test_all  = y.iloc[split:]

# 2. Now apply small/large masks INSIDE train and test sets
small_train_mask = y_train_all <= threshold
large_train_mask = y_train_all > threshold

small_test_mask = y_test_all <= threshold
large_test_mask = y_test_all > threshold

# Final correct splits
X_small_train = X_train_all[small_train_mask]
X_small_test  = X_test_all[small_test_mask]
y_small_train = y_train_all[small_train_mask]
y_small_test  = y_test_all[small_test_mask]

X_large_train = X_train_all[large_train_mask]
X_large_test  = X_test_all[large_test_mask]
y_large_train = y_train_all[large_train_mask]
y_large_test  = y_test_all[large_test_mask]

print(X_small_train.shape)
print(X_large_train.shape)
# =============================================================
# 4. Model A (small & medium spikes)
# =============================================================
model_small = CatBoostRegressor(
    loss_function="Huber:delta=1.0",
    depth=8,
    learning_rate=0.02,
    iterations=4000,
    l2_leaf_reg=3,
    verbose=300
)

model_small.fit(X_small_train, y_small_train,
                eval_set=(X_small_test, y_small_test))

# =============================================================
# 5. Model B (large spikes)
#    → NO log transform
#    → NO quantile loss
#    → No smoothing
# =============================================================
model_large = CatBoostRegressor(
    loss_function="RMSE",
    depth=12,              # deeper to capture nonlinear tail patterns
    learning_rate=0.01,
    iterations=2000,
    l2_leaf_reg=2,
    verbose=300
)

model_large.fit(X_large_train, y_large_train,
                eval_set=(X_large_test, y_large_test))

# =============================================================
# 6. Final prediction function
# =============================================================
def predict_spike_magnitude(X_input):
    # First, decide small vs large region
    # Here: Use model_small to estimate whether the magnitude is in tail
    pred_small = model_small.predict(X_input)

    # If predicted magnitude > threshold, use tail model
    use_large = pred_small > threshold

    preds = np.where(use_large,
                     model_large.predict(X_input),
                     pred_small)

    return preds

# =============================================================
# 7. Make predictions on test set
# =============================================================
final_pred = predict_spike_magnitude(X.iloc[split:])
final_true = y.iloc[split:]

print(pd.DataFrame({
    "true": final_true[:20],
    "pred": final_pred[:20]
}))

import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure arrays
y_true = np.array(final_true)
y_hat = np.array(final_pred)

plt.figure(figsize=(8, 8))

# Scatter points
plt.scatter(y_true, y_hat, alpha=0.6)

# Reference line (perfect prediction)
min_val = min(y_true.min(), y_hat.min())
max_val = max(y_true.max(), y_hat.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Spike Magnitude")
plt.ylabel("Predicted Spike Magnitude")
plt.title("Spike Magnitude Regression: Prediction vs Actual")

plt.grid(True, linestyle="--", alpha=0.4)

# Save to png
plt.savefig("spike_regression_scatter.png", dpi=200)
plt.close()
# # 6. OPTIONAL: Predict higher quantiles
# # ===========================================
# quantiles = [0.9, 0.95]

# q_models = {}
# for q in quantiles:
#     q_params = params.copy()
#     q_params["alpha"] = q
#     q_models[q] = lgb.LGBMRegressor(**q_params)
#     q_models[q].fit(X_train, y_train)
#     print(f"Trained q={q} model")