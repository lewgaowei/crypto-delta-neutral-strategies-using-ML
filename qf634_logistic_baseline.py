#%% [markdown]
# Funding Rate Anomaly Prediction using Logistic Regression
# This script predicts whether the next funding rate will be "extreme negative"
# using Logistic Regression as the baseline classification model.

#%% Imports
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%% Constants
BASE_PATH = Path(__file__).parent

FEATURE_COLS = [
    # Basis features (most important for funding rate prediction)
    'basis', 'basis_30m_avg', 'basis_1h_avg', 'basis_4h_avg',
    'basis_change_30m', 'basis_change_1h',
    'basis_min_30m', 'basis_min_1h',
    'basis_std_30m', 'basis_std_1h', 'basis_std_24h',
    'basis_zscore_30m', 'basis_zscore_1h', 'basis_zscore_24h',

    # Taker imbalance (futures)
    'taker_buy_ratio', 'futures_taker_buy_30m_avg', 'futures_taker_buy_1h_avg',
    'futures_taker_zscore',

    # Taker imbalance (spot)
    'spot_taker_buy_ratio',

    # Futures vs Spot taker difference
    'taker_futures_spot_diff',

    # Price momentum
    'ret_30m', 'ret_1h', 'ret_4h', 'ret_24h',
    'volatility_30m', 'volatility_1h', 'volatility_4h',

    # Volume
    'volume_zscore_30m', 'volume_zscore_1h', 'volume_zscore_4h', 'volume_zscore_24h',
    'trade_count_zscore_30m', 'trade_count_zscore_1h', 'trade_count_zscore_4h', 'trade_count_zscore_24h',

    # Funding rate history (CRITICAL - but only past values!)
    'prev_fr', 'prev_fr_avg_3', 'consecutive_neg',

    # Time features
    'hour_of_day', 'day_of_week'
]

#%% ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_futures_data(symbol, timeframe, base_path=BASE_PATH):
    """Load futures (perpetual) price data."""
    print("="*60)
    print("LOADING FUTURES (PERPS) DATA")
    print("="*60)

    futures_path = base_path / 'raw_historical_price' / f'{symbol}_{timeframe}_binance_futures_historical_data.json'
    df_futures = pd.read_json(futures_path, orient='records', lines=True)

    df_futures['datetime'] = pd.to_datetime(df_futures['datetime'])
    df_futures = df_futures.sort_values('datetime').drop_duplicates(subset=['datetime'])
    df_futures.set_index('datetime', inplace=True)

    futures_numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                            'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in futures_numeric_cols:
        df_futures[col] = df_futures[col].astype(float)

    df_futures = df_futures[futures_numeric_cols]

    print(f"Loaded {len(df_futures)} rows of {symbol} FUTURES data")
    print(f"Date range: {df_futures.index.min()} to {df_futures.index.max()}")

    return df_futures


def load_spot_data(symbol, timeframe, base_path=BASE_PATH):
    """Load spot price data."""
    print("\n" + "="*60)
    print("LOADING SPOT DATA")
    print("="*60)

    spot_path = base_path / 'raw_historical_price' / f'{symbol}_{timeframe}_binance_spot_historical_data.json'
    df_spot = pd.read_json(spot_path, orient='records', lines=True)

    df_spot['datetime'] = pd.to_datetime(df_spot['datetime'])
    df_spot = df_spot.sort_values('datetime').drop_duplicates(subset=['datetime'])
    df_spot.set_index('datetime', inplace=True)

    spot_numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                         'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in spot_numeric_cols:
        df_spot[col] = df_spot[col].astype(float)

    df_spot = df_spot[spot_numeric_cols]
    df_spot = df_spot.rename(columns={col: f'spot_{col}' for col in spot_numeric_cols})

    print(f"Loaded {len(df_spot)} rows of {symbol} SPOT data")
    print(f"Date range: {df_spot.index.min()} to {df_spot.index.max()}")

    return df_spot


def load_funding_rate(symbol, base_path=BASE_PATH):
    """Load funding rate data."""
    print("\n" + "="*60)
    print("LOADING FUNDING RATE DATA")
    print("="*60)

    funding_rate_path = base_path / 'raw_funding_rate' / f'{symbol}_funding_rate_20200101_20251130.csv'
    df_funding = pd.read_csv(funding_rate_path)

    df_funding['datetime'] = pd.to_datetime(df_funding['fundingDateTime'])
    df_funding = df_funding.sort_values('datetime').drop_duplicates(subset=['datetime'])
    df_funding.set_index('datetime', inplace=True)

    df_funding = df_funding[['fundingRate', 'markPrice']]
    df_funding['fundingRate'] = df_funding['fundingRate'].astype(float)
    df_funding['markPrice'] = df_funding['markPrice'].astype(float)

    print(f"Loaded {len(df_funding)} rows of {symbol} FUNDING RATE data")
    print(f"Date range: {df_funding.index.min()} to {df_funding.index.max()}")

    return df_funding


def merge_all_data(df_futures, df_spot, df_funding):
    """Merge futures, spot, and funding rate data."""
    print("\n" + "="*60)
    print("MERGING ALL DATA")
    print("="*60)

    df_merged = df_futures.merge(df_spot, how='inner', left_index=True, right_index=True)
    print(f"After Futures + Spot merge: {len(df_merged)} rows")

    df_merged = df_merged.merge(df_funding, how='left', left_index=True, right_index=True)
    print(f"After adding Funding Rate: {len(df_merged)} rows")
    print(f"Rows with funding rate: {df_merged['fundingRate'].notna().sum()}")

    return df_merged


#%% ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================

def calculate_features(df_merged):
    """Calculate all derived features for ML."""
    print("\n" + "="*60)
    print("CALCULATING DERIVED FEATURES")
    print("="*60)

    df = df_merged.copy()

    # Basis = (Futures - Spot) / Spot * 100 (in percentage)
    df['basis'] = (df['close'] - df['spot_close']) / df['spot_close'] * 100

    # Taker buy ratios
    df['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
    df['spot_taker_buy_ratio'] = df['spot_taker_buy_base_asset_volume'] / df['spot_volume']

    # Returns
    df['returns'] = df['close'].pct_change()
    df['spot_returns'] = df['spot_close'].pct_change()

    # --- 1. BASIS FEATURES ---
    df['basis_30m_avg'] = df['basis'].rolling(6).mean()
    df['basis_1h_avg'] = df['basis'].rolling(12).mean()
    df['basis_4h_avg'] = df['basis'].rolling(48).mean()
    df['basis_change_30m'] = df['basis'] - df['basis'].shift(6)
    df['basis_change_1h'] = df['basis'] - df['basis'].shift(12)
    df['basis_min_30m'] = df['basis'].rolling(6).min()
    df['basis_min_1h'] = df['basis'].rolling(12).min()
    df['basis_std_30m'] = df['basis'].rolling(6).std()
    df['basis_std_1h'] = df['basis'].rolling(12).std()
    df['basis_std_24h'] = df['basis'].rolling(288).std()

    # Z-scores for different timeframes
    basis_30m_mean = df['basis'].rolling(6).mean()
    basis_30m_std = df['basis'].rolling(6).std()
    df['basis_zscore_30m'] = (df['basis'] - basis_30m_mean) / basis_30m_std

    basis_1h_mean = df['basis'].rolling(12).mean()
    basis_1h_std = df['basis'].rolling(12).std()
    df['basis_zscore_1h'] = (df['basis'] - basis_1h_mean) / basis_1h_std

    basis_24h_mean = df['basis'].rolling(288).mean()
    basis_24h_std = df['basis'].rolling(288).std()
    df['basis_zscore_24h'] = (df['basis'] - basis_24h_mean) / basis_24h_std

    # --- 2. TAKER IMBALANCE FEATURES ---
    df['futures_taker_buy_30m_avg'] = df['taker_buy_ratio'].rolling(6).mean()
    df['futures_taker_buy_1h_avg'] = df['taker_buy_ratio'].rolling(12).mean()

    taker_rolling_mean = df['taker_buy_ratio'].rolling(288).mean()
    taker_rolling_std = df['taker_buy_ratio'].rolling(288).std()
    df['futures_taker_zscore'] = (df['taker_buy_ratio'] - taker_rolling_mean) / taker_rolling_std

    df['spot_taker_buy_30m_avg'] = df['spot_taker_buy_ratio'].rolling(6).mean()
    df['spot_taker_buy_1h_avg'] = df['spot_taker_buy_ratio'].rolling(12).mean()

    spot_taker_mean = df['spot_taker_buy_ratio'].rolling(288).mean()
    spot_taker_std = df['spot_taker_buy_ratio'].rolling(288).std()
    df['spot_taker_zscore'] = (df['spot_taker_buy_ratio'] - spot_taker_mean) / spot_taker_std

    df['taker_futures_spot_diff'] = df['taker_buy_ratio'] - df['spot_taker_buy_ratio']

    # --- 3. PRICE MOMENTUM FEATURES ---
    df['ret_30m'] = df['close'].pct_change(6)
    df['ret_1h'] = df['close'].pct_change(12)
    df['ret_4h'] = df['close'].pct_change(48)
    df['ret_24h'] = df['close'].pct_change(288)
    df['volatility_30m'] = df['returns'].rolling(6).std()
    df['volatility_1h'] = df['returns'].rolling(12).std()
    df['volatility_4h'] = df['returns'].rolling(48).std()

    # --- 4. VOLUME FEATURES ---
    df['volume_zscore_30m'] = (df['volume'] - df['volume'].rolling(6).mean()) / df['volume'].rolling(6).std()
    df['volume_zscore_1h'] = (df['volume'] - df['volume'].rolling(12).mean()) / df['volume'].rolling(12).std()
    df['volume_zscore_4h'] = (df['volume'] - df['volume'].rolling(48).mean()) / df['volume'].rolling(48).std()
    df['volume_zscore_24h'] = (df['volume'] - df['volume'].rolling(288).mean()) / df['volume'].rolling(288).std()

    df['trade_count_zscore_30m'] = (df['number_of_trades'] - df['number_of_trades'].rolling(6).mean()) / df['number_of_trades'].rolling(6).std()
    df['trade_count_zscore_1h'] = (df['number_of_trades'] - df['number_of_trades'].rolling(12).mean()) / df['number_of_trades'].rolling(12).std()
    df['trade_count_zscore_4h'] = (df['number_of_trades'] - df['number_of_trades'].rolling(48).mean()) / df['number_of_trades'].rolling(48).std()
    df['trade_count_zscore_24h'] = (df['number_of_trades'] - df['number_of_trades'].rolling(288).mean()) / df['number_of_trades'].rolling(288).std()

    # --- 5. FUNDING RATE HISTORY FEATURES ---
    df['prev_fr'] = df['fundingRate'].ffill()
    df['next_fr'] = df['fundingRate'].bfill()

    funding_only = df[df['fundingRate'].notna()]['fundingRate']
    fr_rolling_avg = funding_only.rolling(3).mean()
    df['prev_fr_avg_3'] = fr_rolling_avg.reindex(df.index).ffill()

    df['is_funding_event'] = df['fundingRate'].notna()

    # Consecutive negative funding count
    def calc_consecutive_neg(series):
        funding_vals = series[series.notna()]
        streaks = []
        current_streak = 0
        for val in funding_vals:
            if val < 0:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return pd.Series(streaks, index=funding_vals.index)

    consec_neg = calc_consecutive_neg(df['fundingRate'])
    df['consecutive_neg'] = consec_neg.reindex(df.index).ffill().fillna(0)

    # --- 6. TIME FEATURES ---
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    print("ML Features calculated successfully!")
    print(f"Total columns: {len(df.columns)}")

    return df


#%% ============================================================
# ML DATA PREPARATION FUNCTIONS
# ============================================================

def prepare_ml_data(df_merged, feature_cols=FEATURE_COLS, fr_percentile=0.25):
    """Prepare ML data with target variable.

    Args:
        df_merged: Merged dataframe with all features
        feature_cols: List of feature columns to use
        fr_percentile: Percentile threshold for "extreme negative" FR (default 0.25 = 25th percentile)
    """
    print("\n" + "="*60)
    print("PREPARING ML DATA")
    print("="*60)

    df = df_merged.copy()

    # Create bars_to_funding
    df['funding_event_group'] = df['is_funding_event'].cumsum()
    df['bars_to_funding'] = df.groupby('funding_event_group').cumcount(ascending=False)

    # Filter to 1 hour to 5 min before funding + funding events
    df_train_candidates = df[
        ((df['bars_to_funding'] >= 1) & (df['bars_to_funding'] <= 12)) |
        (df['is_funding_event'])
    ].copy()

    print(f"Training candidates: {len(df_train_candidates)} rows")

    # Define target variable (percentile = extreme negative)
    funding_rates_only = df[df['fundingRate'].notna()]['fundingRate']
    fr_threshold = funding_rates_only.quantile(fr_percentile)

    print(f"\nFunding Rate Distribution:")
    print(f"  {fr_percentile*100:.0f}th percentile: {fr_threshold*100:.4f}%  <-- THRESHOLD")

    df['target_extreme_neg'] = (df['next_fr'] < fr_threshold).astype(int)
    df_train_candidates['target_extreme_neg'] = df.loc[df_train_candidates.index, 'target_extreme_neg']

    # Check for inf values
    df_temp = df_train_candidates[feature_cols + ['target_extreme_neg']].copy()

    inf_counts = {}
    for col in feature_cols:
        n_inf = np.isinf(df_temp[col]).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf

    if inf_counts:
        print(f"\nWARNING: Columns with inf values:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count} inf values")

    # Replace inf with NaN, then drop
    df_ml = df_temp.replace([np.inf, -np.inf], np.nan)
    df_ml = df_ml.dropna()

    print(f"After dropping NaN/inf: {len(df_ml)} rows")

    return df_ml, df, fr_threshold


def prepare_train_test_split(df_ml, feature_cols=FEATURE_COLS, test_size=0.3):
    """Time-based train/test split."""
    print("\n" + "="*60)
    print("TIME-BASED TRAIN/TEST SPLIT")
    print("="*60)

    split_idx = int(len(df_ml) * (1 - test_size))
    df_train = df_ml.iloc[:split_idx]
    df_test = df_ml.iloc[split_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train['target_extreme_neg']
    X_test = df_test[feature_cols]
    y_test = df_test['target_extreme_neg']

    print(f"Train set: {len(X_train)} samples")
    print(f"  Date range: {df_train.index.min()} to {df_train.index.max()}")
    print(f"  Class 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

    print(f"\nTest set: {len(X_test)} samples")
    print(f"  Date range: {df_test.index.min()} to {df_test.index.max()}")
    print(f"  Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_train, df_test


#%% ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

def train_logistic_baseline(X_train, X_test, y_train, y_test, class_weight_ratio=None):
    """Train Logistic Regression baseline model.

    This is the classification equivalent of linear regression.
    Uses 2D data (samples, features).

    Args:
        X_train: 2D training features (samples, features)
        X_test: 2D test features
        y_train: Training targets
        y_test: Test targets
        class_weight_ratio: Ratio for handling class imbalance (optional)

    Returns:
        model: Trained LogisticRegression model
        results: Dictionary with predictions and metrics
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION BASELINE")
    print("="*60)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # Set up class weights if provided
    if class_weight_ratio is not None:
        class_weight = {0: 1.0, 1: class_weight_ratio}
        print(f"Class weights: {class_weight}")
    else:
        class_weight = 'balanced'
        print("Using balanced class weights")

    # Train Logistic Regression (no regularization for baseline)
    model = LogisticRegression(
        penalty=None,
        max_iter=1000,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("\n--- TRAINING SET ---")
    print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_train, y_train_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_train, y_train_prob):.4f}")

    print("\n--- TEST SET ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Extreme Neg']))

    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])],
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nTop 10 Most Important Features (by |coefficient|):")
    print(feature_importance.head(10).to_string(index=False))

    results = {
        'model': model,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_prob': y_train_prob,
        'y_test_prob': y_test_prob,
        'feature_importance': feature_importance,
        'metrics': {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_auc': roc_auc_score(y_train, y_train_prob),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_prob)
        }
    }

    return model, results


def plot_logistic_results(y_test, y_test_pred, y_test_prob, feature_importance, symbol):
    """Plot Logistic Regression evaluation results."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION VISUALIZATION")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Extreme Neg'],
                yticklabels=['Normal', 'Extreme Neg'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'{symbol} - Logistic Regression Confusion Matrix')

    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{symbol} - ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Feature Importance (Top 15)
    ax3 = axes[1, 0]
    top_features = feature_importance.head(15)
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
    ax3.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'], fontsize=8)
    ax3.set_xlabel('Coefficient')
    ax3.set_title(f'{symbol} - Top 15 Feature Coefficients')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # 4. Probability Distribution
    ax4 = axes[1, 1]
    ax4.hist(y_test_prob[y_test == 0], bins=30, alpha=0.7, label='Normal', color='green', density=True)
    ax4.hist(y_test_prob[y_test == 1], bins=30, alpha=0.7, label='Extreme Neg', color='red', density=True)
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Density')
    ax4.set_title(f'{symbol} - Probability Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


#%% ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def run_logistic_pipeline(symbol, timeframe='5m', fr_percentile=0.25, show_plots=True):
    """Run the full Logistic Regression pipeline for a single symbol.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        timeframe: Candlestick timeframe (default '5m')
        fr_percentile: Percentile threshold for "extreme negative" FR (default 0.25)
        show_plots: Whether to display plots
    """
    print("\n" + "="*80)
    print(f"RUNNING LOGISTIC REGRESSION PIPELINE FOR {symbol}")
    print(f"FR Percentile Threshold: {fr_percentile*100:.0f}th percentile")
    print("="*80)

    # 1. Load Data
    df_futures = load_futures_data(symbol, timeframe)
    df_spot = load_spot_data(symbol, timeframe)
    df_funding = load_funding_rate(symbol)

    # 2. Merge Data
    df_merged = merge_all_data(df_futures, df_spot, df_funding)

    # 3. Calculate Features
    df_merged = calculate_features(df_merged)

    # 4. Prepare ML Data
    df_ml, df_merged, fr_threshold = prepare_ml_data(df_merged, fr_percentile=fr_percentile)

    # 5. Train/Test Split
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_train, df_test = prepare_train_test_split(df_ml)

    # 6. Calculate class weight
    n_class_0 = (y_train == 0).sum()
    n_class_1 = (y_train == 1).sum()
    class_weight_ratio = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0

    print(f"\nClass weight ratio: 1:{class_weight_ratio:.2f}")

    # 7. Train Logistic Regression
    model, results = train_logistic_baseline(
        X_train_scaled, X_test_scaled, y_train, y_test,
        class_weight_ratio=class_weight_ratio
    )

    # 8. Plot Results
    if show_plots:
        plot_logistic_results(
            y_test, results['y_test_pred'], results['y_test_prob'],
            results['feature_importance'], symbol
        )

    # 9. Save Model
    try:
        joblib.dump(model, BASE_PATH / f'{symbol}_logistic_model.joblib')
        joblib.dump(scaler, BASE_PATH / f'{symbol}_scaler.joblib')
        print(f"\nModel saved to {symbol}_logistic_model.joblib")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")

    print("\n" + "="*60)
    print(f"LOGISTIC REGRESSION PIPELINE COMPLETE FOR {symbol}")
    print("="*60)

    return {
        'model': model,
        'scaler': scaler,
        'results': results,
        'fr_threshold': fr_threshold,
        'df_merged': df_merged
    }


#%% ============================================================
# RUN SINGLE SYMBOL
# ============================================================

SYMBOL = 'RESOLVUSDT'
TIMEFRAME = '5m'
FR_PERCENTILE = 0.25

results = run_logistic_pipeline(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    fr_percentile=FR_PERCENTILE,
    show_plots=True
)

# Print final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Symbol: {SYMBOL}")
print(f"Test Accuracy:  {results['results']['metrics']['test_accuracy']:.4f}")
print(f"Test Precision: {results['results']['metrics']['test_precision']:.4f}")
print(f"Test Recall:    {results['results']['metrics']['test_recall']:.4f}")
print(f"Test F1:        {results['results']['metrics']['test_f1']:.4f}")
print(f"Test AUC:       {results['results']['metrics']['test_auc']:.4f}")
