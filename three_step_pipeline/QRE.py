import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

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

df = df_prepared_features.copy()

df_spike = df[df["spike_label"] == 1].copy()
df_spike["abs_mag"] = df_spike[FUNDING_COL].abs()

y = df_spike["abs_mag"]
X = df_spike.drop(columns=["spike_label",FUNDING_COL, "abs_mag"])

# ============================================================
# 2. Time-based split (80% train / 20% test)
# ============================================================
split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]

# ============================================================
# 3. Train CatBoost Quantile models for multiple alphas
# ============================================================
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
q_models = {}

for q in quantiles:
    print(f"\nTraining Quantile model, alpha={q}...")
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        depth=8,
        learning_rate=0.03,
        iterations=2000,
        l2_leaf_reg=4,
        random_seed=42,
        verbose=300
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    q_models[q] = model

print("\nAll quantile models trained.")

# ============================================================
# 4. Prediction helper (ensemble over quantiles)
# ============================================================
def predict_abs_mag_qre(features, q_models, quantiles,
                        weights=None):
    """
    QRE prediction for absolute magnitude.

    features : pd.DataFrame
    q_models : dict[alpha -> CatBoostRegressor]
    quantiles: list of alphas (same order used in training)
    weights  : list of weights (same length as quantiles), optional

    Returns:
        - preds_all : (n_samples, n_quantiles) matrix
        - pred_final: 1D final magnitude prediction
    """
    preds_all = np.vstack([
        q_models[q].predict(features) for q in quantiles
    ]).T  # shape: (n_samples, n_q)

    if weights is None:
        # Example: emphasize median & upper tail
        # (tweak these numbers as you like)
        w_dict = {
            0.1: 0.05,
            0.3: 0.10,
            0.5: 0.40,
            0.7: 0.25,
            0.9: 0.15,
            0.95: 0.05
        }
        weights = np.array([w_dict[q] for q in quantiles])
    else:
        weights = np.array(weights)

    weights = weights / weights.sum()
    pred_final = (preds_all * weights).sum(axis=1)

    return preds_all, pred_final

# Make predictions on test set
preds_all_test, pred_final_test = predict_abs_mag_qre(
    X_test, q_models, quantiles
)

# ============================================================
# 5. Scatter plot: Predicted vs True ABS magnitude
# ============================================================
plt.figure(figsize=(8, 8))
plt.scatter(y_test.values, pred_final_test, alpha=0.7, s=40)

min_val = min(y_test.min(), pred_final_test.min())
max_val = max(y_test.max(), pred_final_test.max())
plt.plot([min_val, max_val], [min_val, max_val],
         linestyle="--", color="gray", linewidth=2)

plt.title("Quantile Regression Ensemble — Predicted vs True |Spike Magnitude|",
          fontsize=14)
plt.xlabel("True |Spike Magnitude|")
plt.ylabel("Predicted |Spike Magnitude|")
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig("spike_regression_scatter.png", dpi=200)
plt.close()

# ============================================================
# 6. Quick sanity check
# ============================================================
print("\nSample predictions:")
print(pd.DataFrame({
    "true_abs_mag": y_test.values[:20],
    "pred_abs_mag": pred_final_test[:20]
}))