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
df_prepared_features = df.dropna()

# ======================================================
# 7. SPIKE LABEL CREATION
# ======================================================
SPIKE_Q = 0.90
spike_threshold = df_prepared_features[FUNDING_COL].abs().quantile(SPIKE_Q)
df_prepared_features["spike_label"] = (df_prepared_features[FUNDING_COL].abs() >= spike_threshold).astype(int)
# drop_cols = [FUNDING_COL, "is_spike"]
# feature_cols = [c for c in df.columns if c not in drop_cols]

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

# =============================================================
# 1. Load dataset (must contain spike_label, spike_magnitude)
# =============================================================
df = df_prepared_features.copy()

df_spike = df[df["spike_label"] == 1].copy()
y_mag = df_spike[FUNDING_COL].abs()
X = df_spike.drop(columns=["spike_label", FUNDING_COL])

# =============================================================
# 2. Compute small/large threshold and create classifier target
# =============================================================
threshold = y_mag.quantile(0.25)
df_spike["size_label"] = (y_mag > threshold).astype(int)

y_cls = df_spike["size_label"]
X_cls = X.copy()

# =============================================================
# 3. Time-based split (80/20)
# =============================================================
split = int(len(X_cls) * 0.8)

X_train_cls = X_cls.iloc[:split]
X_test_cls  = X_cls.iloc[split:]

y_train_cls = y_cls.iloc[:split]
y_test_cls  = y_cls.iloc[split:]

# =============================================================
# 4. Stage 1: Train Small-vs-Large Classifier
# =============================================================
cls_model = CatBoostClassifier(
    loss_function="Logloss",
    depth=8,
    learning_rate=0.03,
    iterations=2000,
    random_seed=42,
    verbose=300
)
cls_model.fit(X_train_cls, y_train_cls, eval_set=(X_test_cls, y_test_cls))

# =============================================================
# 5. Stage 2: Split magnitude dataset INTO TRAIN SMALL/LARGE
#    IMPORTANT: Split AFTER classifier training split, not before
# =============================================================
X_mag_train = X.iloc[:split]
X_mag_test  = X.iloc[split:]

y_mag_train = y_mag.iloc[:split]
y_mag_test  = y_mag.iloc[split:]

# Partition TRAIN set by true size label
small_mask_train = y_mag_train <= threshold
large_mask_train = y_mag_train > threshold

X_small_train = X_mag_train[small_mask_train]
y_small_train = y_mag_train[small_mask_train]

X_large_train = X_mag_train[large_mask_train]
y_large_train = y_mag_train[large_mask_train]

# Partition TEST set by true size label (for evaluation only)
small_mask_test = y_mag_test <= threshold
large_mask_test = y_mag_test > threshold

X_small_test = X_mag_test[small_mask_test]
y_small_test = y_mag_test[small_mask_test]

X_large_test = X_mag_test[large_mask_test]
y_large_test = y_mag_test[large_mask_test]

print("Train shapes:")
print("Small:", X_small_train.shape, " Large:", X_large_train.shape)

# =============================================================
# 6. Train Small-Magnitude Model
# =============================================================
model_small = CatBoostRegressor(
    loss_function="Huber:delta=1.0",
    depth=8,
    learning_rate=0.02,
    iterations=3000,
    verbose=300
)

model_small.fit(
    X_small_train, y_small_train,
    eval_set=(X_small_test, y_small_test)
)

# =============================================================
# 7. Train Large-Magnitude Model
# =============================================================
if len(y_large_train) > 0:
    model_large = CatBoostRegressor(
        loss_function="RMSE",
        depth=12,
        learning_rate=0.01,
        iterations=5000,
        verbose=300
    )

    model_large.fit(
        X_large_train, y_large_train,
        eval_set=(X_large_test, y_large_test)
    )
else:
    model_large = None
    print("⚠ WARNING: No large spikes in training dataset.")

# =============================================================
# 8. Combined Predict Function
# =============================================================
def predict_magnitude(features):
    """
    Predict spike magnitude using:
      1. Classifier → small or large
      2. Small/large regressor accordingly
    """

    # Stage 1: Predict probability of large spike
    size_prob = cls_model.predict_proba(features)[:, 1]
    size_label = (size_prob > 0.5).astype(int)

    # Stage 2: Compute magnitude predictions
    small_pred = model_small.predict(features)

    if model_large is not None:
        large_pred = model_large.predict(features)
    else:
        large_pred = small_pred  # fallback

    magnitude_pred = np.where(size_label == 1, large_pred, small_pred)

    return {
        "size_prob": size_prob,
        "size_label": size_label,
        "magnitude_pred": magnitude_pred
    }

print("\nPipeline Ready!")

result = predict_magnitude(X_mag_test)

pred = result["magnitude_pred"]
cls_labels = result["size_label"]
cls_probs = result["size_prob"]

import matplotlib.pyplot as plt

def plot_magnitude_scatter(y_true, result):
    """
    Scatter plot of predicted vs true spike magnitude.

    Parameters
    ----------
    y_true : array-like
        Actual spike magnitudes
    result : dict
        Output from predict_spike_magnitude(), must contain:
          - result["magnitude_pred"]
          - result["size_label"]
    """

    pred = result["magnitude_pred"]
    size_label = result["size_label"]  # 0 = small, 1 = large

    # Color map: small = blue, large = red
    colors = np.where(size_label == 1, "red", "blue")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, pred, c=colors, alpha=0.7, s=50)

    # Diagonal reference line
    min_val = min(min(y_true), min(pred))
    max_val = max(max(y_true), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             "k--", lw=2, alpha=0.7)

    plt.title("Spike Magnitude Prediction — Scatter Plot", fontsize=15)
    plt.xlabel("True Spike Magnitude")
    plt.ylabel("Predicted Spike Magnitude")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Legend for colors
    import matplotlib.patches as mpatches
    small_patch = mpatches.Patch(color="blue", label="Predicted Small Spike")
    large_patch = mpatches.Patch(color="red", label="Predicted Large Spike")
    plt.legend(handles=[small_patch, large_patch])

    plt.savefig("spike_regression_scatter.png", dpi=200)
    plt.close()


plot_magnitude_scatter(y_mag_test.values, result)