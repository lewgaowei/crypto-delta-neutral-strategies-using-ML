import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


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

# ==========================================================
# 3. REMOVE LAST 15 MINUTES BEFORE EACH FUNDING TIMESTAMP
# ==========================================================
fund_times = funding.index.sort_values()

clean_spot_windows = {}
clean_future_windows = {}

for i in range(1, len(fund_times)):
    t_prev = fund_times[i - 1]
    t_curr = fund_times[i]

    window_start = t_prev
    window_end = t_curr - pd.Timedelta(minutes=15)  # remove final 15m

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

# --------------------------------------------------
# 0. ASSUME df IS ALREADY BUILT FROM YOUR CODE
#     df = funding.merge(spot_8h, ...) ...
# --------------------------------------------------

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

y = df["is_spike"]

# ======================================================
# 8. FEATURE MATRIX
# ======================================================
drop_cols = [FUNDING_COL, "is_spike"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]

# (Optional) handle missing values
# LightGBM can handle NaNs directly, but if you prefer:
# X = X.fillna(0)

# ==========================================================
# 3. TIME-BASED TRAIN / TEST SPLIT (70% / 30%)
# ==========================================================
n_total = len(df)
split_idx = int(n_total * 0.7)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("Train spike ratio:", y_train.mean())
print("Test spike ratio:", y_test.mean())

# ==========================================================
# 4. HANDLE CLASS IMBALANCE (scale_pos_weight)
# ==========================================================
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
if n_pos == 0:
    scale_pos_weight = 1.0
else:
    scale_pos_weight = n_neg / n_pos

print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# ==========================================================
# 5. TRAIN LIGHTGBM CLASSIFIER
# ==========================================================
lgb_clf = LGBMClassifier(
    objective="binary",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # key for rare spikes
    random_state=42,
    n_jobs=-1,
)

lgb_clf.fit(X_train, y_train)

# ==========================================================
# 6. EVALUATION: PROBA, METRICS, CONFUSION MATRIX
# ==========================================================
y_proba = lgb_clf.predict_proba(X_test)[:, 1]

# Default threshold 0.5 (you will likely want to LOWER this for more spike recall)
THRESH = 0.5
y_pred = (y_proba >= THRESH).astype(int)

print("\nConfusion matrix (threshold = {:.2f}):".format(THRESH))
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")
except ValueError:
    print("ROC AUC cannot be computed (check if both classes exist in y_test).")

# ==========================================================
# 7. OPTIONAL: ADJUST THRESHOLD FOR BETTER SPIKE RECALL
# ==========================================================
for thr in [0.3, 0.2, 0.1]:
    y_pred_thr = (y_proba >= thr).astype(int)
    print(f"\n=== Threshold = {thr:.2f} ===")
    print(confusion_matrix(y_test, y_pred_thr))
    print(classification_report(y_test, y_pred_thr, digits=4))