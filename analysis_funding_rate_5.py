#%% [markdown]
# Funding Rate Direction Prediction using Machine Learning
# This script predicts whether the next funding rate will be positive or negative

#%% Imports
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%% Configuration
# === USER CONFIGURATION ===
SYMBOL = 'SUPERUSDT'  # Change to any symbol: 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', etc.
RANDOM_STATE = 42

# Rolling/Recursive Training Configuration
PREDICTION_WINDOW_DAYS = 7  # Predict 1 week at a time
PERIODS_PER_DAY = 3         # Funding every 8 hours
PREDICTION_WINDOW = PREDICTION_WINDOW_DAYS * PERIODS_PER_DAY  # 21 periods per week
MIN_TRAIN_WEEKS = 12        # Minimum training data (weeks) before first prediction

# Data directory
DATA_DIR = Path(__file__).parent / "raw_funding_rate"
if not DATA_DIR.exists():
    DATA_DIR = Path("raw_funding_rate")

print(f"Selected Symbol: {SYMBOL}")
print(f"Prediction Window: {PREDICTION_WINDOW_DAYS} days ({PREDICTION_WINDOW} periods)")
print(f"Min Training Data: {MIN_TRAIN_WEEKS} weeks")

#%% Load Data
def load_funding_data(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load funding rate data for a specific symbol."""
    csv_files = list(data_dir.glob(f"{symbol}_funding_rate_*.csv"))

    if not csv_files:
        available = [f.name.split("_")[0] for f in data_dir.glob("*_funding_rate_*.csv")]
        raise FileNotFoundError(
            f"No data found for {symbol}. Available symbols: {available}"
        )

    df = pd.read_csv(csv_files[0])
    df['fundingDateTime'] = pd.to_datetime(df['fundingDateTime'])
    df = df.sort_values('fundingDateTime').reset_index(drop=True)

    print(f"Loaded {len(df):,} records for {symbol}")
    print(f"Date range: {df['fundingDateTime'].min()} to {df['fundingDateTime'].max()}")

    return df

df = load_funding_data(DATA_DIR, SYMBOL)

#%% Feature Engineering
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from funding rate data."""
    df = df.copy()

    # Lag features (previous funding rates)
    for i in range(1, 13):
        df[f'lag_{i}'] = df['fundingRate'].shift(i)

    # Rolling statistics
    df['rolling_mean_6'] = df['fundingRate'].rolling(window=6).mean()
    df['rolling_mean_12'] = df['fundingRate'].rolling(window=12).mean()
    df['rolling_mean_24'] = df['fundingRate'].rolling(window=24).mean()

    df['rolling_std_6'] = df['fundingRate'].rolling(window=6).std()
    df['rolling_std_12'] = df['fundingRate'].rolling(window=12).std()

    df['rolling_min_6'] = df['fundingRate'].rolling(window=6).min()
    df['rolling_max_6'] = df['fundingRate'].rolling(window=6).max()

    # Momentum features
    df['momentum_3'] = df['fundingRate'] - df['fundingRate'].shift(3)
    df['momentum_6'] = df['fundingRate'] - df['fundingRate'].shift(6)

    # Rate of change
    df['rate_change'] = df['fundingRate'].pct_change()

    # Acceleration (second derivative)
    df['acceleration'] = df['momentum_3'] - df['momentum_3'].shift(1)

    # Streak: consecutive positive/negative funding rates
    df['is_positive'] = (df['fundingRate'] > 0).astype(int)
    streak = []
    count = 0
    prev_sign = None
    for val in df['is_positive']:
        if prev_sign is None:
            count = 1
        elif val == prev_sign:
            count += 1
        else:
            count = 1
        streak.append(count if val == 1 else -count)
        prev_sign = val
    df['streak'] = streak

    # Time-based features
    df['hour'] = df['fundingDateTime'].dt.hour
    df['day_of_week'] = df['fundingDateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['fundingDateTime'].dt.month

    # Distance from neutral rate (0.01% = 0.0001)
    neutral_rate = 0.0001
    df['dist_from_neutral'] = df['fundingRate'] - neutral_rate

    return df

df_features = create_features(df)
print(f"\nCreated {len([c for c in df_features.columns if c not in df.columns])} new features")

#%% Create Target Variable
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variable: Will NEXT funding rate be positive?"""
    df = df.copy()
    # Target: 1 if next funding rate is positive, 0 otherwise
    df['target'] = (df['fundingRate'].shift(-1) > 0).astype(int)
    return df

df_ml = create_target(df_features)

# Show original dataframe before dropping NaN
print(f"\nOriginal records: {len(df_ml):,}")
print(f"Date range: {df_ml['fundingDateTime'].min()} to {df_ml['fundingDateTime'].max()}")

# Identify rows with NaN
nan_rows = df_ml[df_ml.isnull().any(axis=1)]
print(f"\nRows with NaN (will be dropped): {len(nan_rows)}")
print("Dropped dates:")
print(f"  Start: {nan_rows['fundingDateTime'].min()}")
print(f"  End:   {nan_rows['fundingDateTime'].max()}")
print(f"\nFirst 15 dropped rows:")
print(nan_rows[['fundingDateTime', 'fundingRate']].head(15).to_string(index=False))

# Drop rows with NaN (from lag features and target)
df_ml = df_ml.dropna()
print(f"\nRecords after dropping NaN: {len(df_ml):,}")
print(f"Remaining date range: {df_ml['fundingDateTime'].min()} to {df_ml['fundingDateTime'].max()}")

# Check class balance
pos_ratio = df_ml['target'].mean()
print(f"Class balance: {pos_ratio*100:.1f}% positive, {(1-pos_ratio)*100:.1f}% negative")

#%% Prepare Features and Target
# Define feature columns (exclude non-feature columns)
exclude_cols = ['symbol', 'fundingRate', 'markPrice', 'fundingDateTime',
                'formattedFundingDateTime', 'target', 'is_positive']
feature_cols = [c for c in df_ml.columns if c not in exclude_cols]

print(f"\nFeatures ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

X = df_ml[feature_cols].values
y = df_ml['target'].values

#%% Rolling/Recursive Walk-Forward Training
# Train on all data up to each week, then predict the next week
# Retrain model each week with expanding window

MIN_TRAIN_SIZE = MIN_TRAIN_WEEKS * PREDICTION_WINDOW  # Minimum training samples

print(f"\n{'='*70}")
print("RECURSIVE WALK-FORWARD TRAINING")
print(f"{'='*70}")
print(f"Min training size: {MIN_TRAIN_SIZE} samples ({MIN_TRAIN_WEEKS} weeks)")
print(f"Prediction window: {PREDICTION_WINDOW} samples ({PREDICTION_WINDOW_DAYS} days)")

# Calculate number of prediction windows
n_windows = (len(X) - MIN_TRAIN_SIZE) // PREDICTION_WINDOW
print(f"Number of prediction windows: {n_windows}")

#%% Define model factory functions
def get_model(name):
    """Factory function to create fresh model instances."""
    if name == 'Logistic Regression':
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    elif name == 'SVM':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    elif name == 'XGBoost' and HAS_XGBOOST:
        return XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             use_label_encoder=False, eval_metric='logloss')
    return None

model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM']
if HAS_XGBOOST:
    model_names.append('XGBoost')

print(f"\nModels to train: {model_names}")

#%% Run Walk-Forward Validation
# Store all predictions for each model
all_predictions = {name: {'y_true': [], 'y_pred': [], 'y_prob': [], 'dates': [], 'window': []}
                   for name in model_names}

# Track per-window performance
window_results = []

for window_idx in range(n_windows):
    # Define train and test indices for this window
    train_end = MIN_TRAIN_SIZE + (window_idx * PREDICTION_WINDOW)
    test_start = train_end
    test_end = min(test_start + PREDICTION_WINDOW, len(X))

    # Get train and test data
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]

    # Get dates for this window
    train_start_date = df_ml['fundingDateTime'].iloc[0]
    train_end_date = df_ml['fundingDateTime'].iloc[train_end - 1]
    test_start_date = df_ml['fundingDateTime'].iloc[test_start]
    test_end_date = df_ml['fundingDateTime'].iloc[test_end - 1]

    if window_idx % 10 == 0:  # Print progress every 10 windows
        print(f"\nWindow {window_idx + 1}/{n_windows}")
        print(f"  Train: {train_start_date.date()} to {train_end_date.date()} ({len(X_train)} samples)")
        print(f"  Test:  {test_start_date.date()} to {test_end_date.date()} ({len(X_test)} samples)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    window_metrics = {'window': window_idx + 1, 'test_start': test_start_date, 'test_end': test_end_date}

    # Train each model on this window
    for name in model_names:
        model = get_model(name)
        if model is None:
            continue

        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # Store predictions
        all_predictions[name]['y_true'].extend(y_test)
        all_predictions[name]['y_pred'].extend(y_pred)
        all_predictions[name]['y_prob'].extend(y_prob)
        all_predictions[name]['dates'].extend(df_ml['fundingDateTime'].iloc[test_start:test_end].values)
        all_predictions[name]['window'].extend([window_idx + 1] * len(y_test))

        # Calculate window accuracy
        window_metrics[f'{name}_acc'] = accuracy_score(y_test, y_pred)

    window_results.append(window_metrics)

print(f"\n\nCompleted {n_windows} prediction windows!")

#%% Calculate Overall Results
results = {}

for name in model_names:
    y_true = np.array(all_predictions[name]['y_true'])
    y_pred = np.array(all_predictions[name]['y_pred'])
    y_prob = np.array(all_predictions[name]['y_prob'])

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results[name] = {
        'accuracy': acc,
        'roc_auc': roc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_true': y_true,
        'dates': all_predictions[name]['dates'],
        'windows': all_predictions[name]['window']
    }

    print(f"\n{name}:")
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  Overall ROC-AUC:  {roc:.4f}")
    print(f"  Overall F1 Score: {f1:.4f}")

# Create y_test for compatibility with later cells
y_test = results[model_names[0]]['y_true']

#%% Model Comparison Summary
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

summary_df = pd.DataFrame({
    name: {
        'Accuracy': f"{r['accuracy']:.4f}",
        'ROC-AUC': f"{r['roc_auc']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1 Score': f"{r['f1']:.4f}"
    }
    for name, r in results.items()
}).T

print(summary_df.to_string())

# Find best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
print(f"\nBest Model (by ROC-AUC): {best_model_name}")

#%% Confusion Matrix Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    if idx < len(axes):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {r["accuracy"]:.4f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

# Hide empty subplot if odd number of models
if len(results) < len(axes):
    for i in range(len(results), len(axes)):
        axes[i].axis('off')

plt.tight_layout()
plt.suptitle(f'Confusion Matrices - {SYMBOL}', y=1.02, fontsize=14)
plt.show()

#%% ROC Curves
plt.figure(figsize=(10, 8))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves - {SYMBOL} Funding Rate Direction Prediction')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

#%% Feature Importance (Train final models on all data)
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)
print("Training final models on all data for feature importance...")

# Train Random Forest on all data for feature importance
rf_final = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_final.fit(X, y)

feat_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feat_importance['feature'][:15], feat_importance['importance'][:15])
plt.xlabel('Feature Importance')
plt.title(f'Top 15 Features - Random Forest ({SYMBOL})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 Features (Random Forest):")
print(feat_importance.head(10).to_string(index=False))

#%% Feature Importance (XGBoost)
if HAS_XGBOOST:
    xgb_final = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                               use_label_encoder=False, eval_metric='logloss')
    xgb_final.fit(X, y)

    feat_importance_xgb = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_final.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(feat_importance_xgb['feature'][:15], feat_importance_xgb['importance'][:15])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Features - XGBoost ({SYMBOL})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("\nTop 10 Features (XGBoost):")
    print(feat_importance_xgb.head(10).to_string(index=False))

#%% Per-Window Accuracy Over Time
print("\n" + "="*70)
print("PER-WINDOW ACCURACY ANALYSIS")
print("="*70)

window_df = pd.DataFrame(window_results)
print(f"\nAccuracy per prediction window (each = {PREDICTION_WINDOW_DAYS} days):")
print(window_df.to_string(index=False))

# Plot per-window accuracy over time
fig, ax = plt.subplots(figsize=(14, 6))

for name in model_names:
    acc_col = f'{name}_acc'
    if acc_col in window_df.columns:
        ax.plot(window_df['window'], window_df[acc_col], label=name, marker='o', markersize=3, alpha=0.7)

ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax.set_xlabel('Prediction Window')
ax.set_ylabel('Accuracy')
ax.set_title(f'Walk-Forward Accuracy per {PREDICTION_WINDOW_DAYS}-Day Window - {SYMBOL}')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary statistics per model
print("\nPer-Window Accuracy Statistics:")
for name in model_names:
    acc_col = f'{name}_acc'
    if acc_col in window_df.columns:
        accs = window_df[acc_col]
        print(f"  {name}: Mean={accs.mean():.4f}, Std={accs.std():.4f}, Min={accs.min():.4f}, Max={accs.max():.4f}")

#%% Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall walk-forward performance
metrics_df = pd.DataFrame({
    name: [r['accuracy'], r['roc_auc'], r['f1']]
    for name, r in results.items()
}, index=['Accuracy', 'ROC-AUC', 'F1 Score'])

metrics_df.T.plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_title(f'Walk-Forward Overall Performance - {SYMBOL}')
axes[0].set_ylabel('Score')
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1)

# Per-window accuracy distribution (box plot)
window_accs = {name: window_df[f'{name}_acc'].values for name in model_names if f'{name}_acc' in window_df.columns}
axes[1].boxplot(window_accs.values(), labels=window_accs.keys())
axes[1].set_title(f'Per-Window Accuracy Distribution - {SYMBOL}')
axes[1].set_ylabel('Accuracy')
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%% Prediction Timeline
best_result = results[best_model_name]
test_dates = np.array(best_result['dates'])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Actual vs Predicted
axes[0].plot(test_dates, best_result['y_true'], label='Actual', alpha=0.7)
axes[0].plot(test_dates, best_result['y_pred'], label='Predicted', alpha=0.7, linestyle='--')
axes[0].set_ylabel('Direction (0=Neg, 1=Pos)')
axes[0].set_title(f'{best_model_name} Walk-Forward Predictions vs Actual - {SYMBOL}')
axes[0].legend()

# Prediction probability
axes[1].plot(test_dates, best_result['y_prob'], label='Probability of Positive', color='green')
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
axes[1].fill_between(test_dates, 0, best_result['y_prob'], alpha=0.3)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Probability')
axes[1].set_title('Prediction Confidence')
axes[1].legend()

plt.tight_layout()
plt.show()

#%% Classification Report
print("\n" + "="*70)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("="*70)
print(classification_report(best_result['y_true'], best_result['y_pred'],
                            target_names=['Negative', 'Positive']))

#%% Summary
print("\n" + "="*70)
print("WALK-FORWARD VALIDATION SUMMARY")
print("="*70)
print(f"""
Symbol: {SYMBOL}
Total Records: {len(df_ml):,}
Features Used: {len(feature_cols)}

Walk-Forward Configuration:
  - Min Training Data: {MIN_TRAIN_WEEKS} weeks ({MIN_TRAIN_SIZE} samples)
  - Prediction Window: {PREDICTION_WINDOW_DAYS} days ({PREDICTION_WINDOW} samples)
  - Number of Windows: {n_windows}
  - Total Predictions: {len(best_result['y_true']):,}

Best Model: {best_model_name}
  - Overall Accuracy: {results[best_model_name]['accuracy']:.4f}
  - Overall ROC-AUC: {results[best_model_name]['roc_auc']:.4f}
  - Overall F1 Score: {results[best_model_name]['f1']:.4f}

Class Distribution (in predictions):
  - Positive rates: {sum(best_result['y_true']):,} ({np.mean(best_result['y_true'])*100:.1f}%)
  - Negative rates: {len(best_result['y_true']) - sum(best_result['y_true']):,} ({(1-np.mean(best_result['y_true']))*100:.1f}%)
""")

print("\nTo change the symbol, modify SYMBOL at the top of this script.")
print("Available symbols can be found in the raw_funding_rate folder.")

#%% Final Results DataFrame
# Create DataFrame with actual vs predicted directions from walk-forward validation
best_result = results[best_model_name]

results_df = pd.DataFrame({
    'datetime': best_result['dates'],
    'window': best_result['windows'],
    'actual_direction': best_result['y_true'],
    'predicted_direction': best_result['y_pred'],
    'prediction_prob': best_result['y_prob'],
    'correct': (np.array(best_result['y_true']) == np.array(best_result['y_pred'])).astype(int)
})

# Map direction to readable labels
results_df['actual_label'] = results_df['actual_direction'].map({1: 'Positive', 0: 'Negative'})
results_df['predicted_label'] = results_df['predicted_direction'].map({1: 'Positive', 0: 'Negative'})

print(f"\n{'='*70}")
print(f"WALK-FORWARD PREDICTION RESULTS - {best_model_name} on {SYMBOL}")
print(f"{'='*70}")
print(f"\nTotal predictions: {len(results_df)}")
print(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.1f}%)")

print("\nFirst 20 predictions:")
print(results_df[['datetime', 'window', 'actual_label', 'predicted_label', 'prediction_prob', 'correct']].head(20).to_string(index=False))

print("\nLast 20 predictions:")
print(results_df[['datetime', 'window', 'actual_label', 'predicted_label', 'prediction_prob', 'correct']].tail(20).to_string(index=False))

# Export to CSV
results_df.to_csv(f'{SYMBOL}_walkforward_predictions.csv', index=False)
print(f"\nResults exported to: {SYMBOL}_walkforward_predictions.csv")

# Also export window results
window_df.to_csv(f'{SYMBOL}_window_accuracy.csv', index=False)
print(f"Window accuracy exported to: {SYMBOL}_window_accuracy.csv")

#%% ============================================================================
# PROFITABILITY ANALYSIS
# ============================================================================

#%% Trading Configuration
# === TRADING COSTS ===
ENTRY_FEE = 0.001       # 0.1% entry fee (spot buy + perp short)
EXIT_FEE = 0.001        # 0.1% exit fee (spot sell + perp close)
SLIPPAGE = 0.0001       # 0.01% slippage per trade
TOTAL_COST = ENTRY_FEE + EXIT_FEE + (SLIPPAGE * 2)  # 0.22% total round-trip

print("\n" + "="*70)
print("PROFITABILITY ANALYSIS")
print("="*70)
print(f"""
Trading Strategy: Long Spot + Short Perpetual (Delta-Neutral)
  - Entry Fee: {ENTRY_FEE*100:.2f}%
  - Exit Fee: {EXIT_FEE*100:.2f}%
  - Slippage: {SLIPPAGE*100:.3f}% per trade
  - Total Cost (round-trip): {TOTAL_COST*100:.2f}%
""")

#%% Consecutive Positive Streak Analysis (Historical)
def analyze_streaks(funding_rates):
    """Analyze consecutive positive funding rate streaks."""
    streaks = []
    current_streak = 0
    streak_rates = []

    for rate in funding_rates:
        if rate > 0:
            current_streak += 1
            streak_rates.append(rate)
        else:
            if current_streak > 0:
                streaks.append({
                    'length': current_streak,
                    'total_rate': sum(streak_rates),
                    'avg_rate': np.mean(streak_rates)
                })
            current_streak = 0
            streak_rates = []

    # Don't forget the last streak
    if current_streak > 0:
        streaks.append({
            'length': current_streak,
            'total_rate': sum(streak_rates),
            'avg_rate': np.mean(streak_rates)
        })

    return streaks

# Analyze historical streaks from raw data
historical_streaks = analyze_streaks(df['fundingRate'].values)
streak_df = pd.DataFrame(historical_streaks)

print("\n" + "-"*50)
print("HISTORICAL STREAK ANALYSIS")
print("-"*50)

# Group by streak length
streak_summary = streak_df.groupby('length').agg({
    'total_rate': ['count', 'mean', 'sum'],
    'avg_rate': 'mean'
}).round(6)
streak_summary.columns = ['count', 'avg_total_rate', 'sum_total_rate', 'avg_period_rate']
streak_summary = streak_summary.reset_index()

# Calculate profitability for each streak length
streak_summary['gross_return'] = streak_summary['avg_total_rate'] * 100
streak_summary['net_return'] = (streak_summary['avg_total_rate'] - TOTAL_COST) * 100
streak_summary['profitable'] = streak_summary['net_return'] > 0

print("\nStreak Length Distribution:")
print(streak_summary.to_string(index=False))

#%% Breakeven Calculator
def calculate_breakeven(avg_rate_per_period, total_cost):
    """Calculate number of periods needed to breakeven."""
    if avg_rate_per_period <= 0:
        return float('inf')
    return total_cost / avg_rate_per_period

# Calculate average positive funding rate
positive_rates = df[df['fundingRate'] > 0]['fundingRate']
avg_positive_rate = positive_rates.mean()
median_positive_rate = positive_rates.median()

breakeven_avg = calculate_breakeven(avg_positive_rate, TOTAL_COST)
breakeven_median = calculate_breakeven(median_positive_rate, TOTAL_COST)

print("\n" + "-"*50)
print("BREAKEVEN ANALYSIS")
print("-"*50)
print(f"""
Positive Funding Rate Statistics:
  - Mean: {avg_positive_rate*100:.4f}%
  - Median: {median_positive_rate*100:.4f}%
  - Count: {len(positive_rates):,} periods ({len(positive_rates)/len(df)*100:.1f}% of all)

Breakeven Periods Needed:
  - Using Mean Rate: {breakeven_avg:.1f} periods ({breakeven_avg/3:.1f} days)
  - Using Median Rate: {breakeven_median:.1f} periods ({breakeven_median/3:.1f} days)
""")

# Find minimum profitable streak length
min_profitable_streak = streak_summary[streak_summary['profitable']]['length'].min()
if pd.isna(min_profitable_streak):
    print("WARNING: No streak length is profitable with current fee structure!")
else:
    print(f"Minimum Profitable Streak Length: {min_profitable_streak} periods ({min_profitable_streak/3:.1f} days)")

#%% Strategy Backtester
def backtest_strategy(predictions_df, actual_rates_df, total_cost, min_streak=1):
    """
    Backtest the funding rate strategy using ML predictions.

    Strategy:
    - Enter (long spot + short perp) when model predicts positive
    - Exit when model predicts negative
    - Pay fees/slippage on entry and exit
    """
    trades = []
    in_position = False
    entry_idx = None
    collected_rates = []

    # Merge predictions with actual rates
    pred_dates = set(pd.to_datetime(predictions_df['datetime']))

    for i, row in predictions_df.iterrows():
        pred = row['predicted_direction']
        actual = row['actual_direction']

        # Get actual funding rate for this period
        # actual_direction tells us if NEXT rate is positive
        # We need the actual rate value
        dt = pd.to_datetime(row['datetime'])
        rate_row = actual_rates_df[actual_rates_df['fundingDateTime'] == dt]
        if len(rate_row) == 0:
            continue
        actual_rate = rate_row['fundingRate'].values[0]

        if not in_position and pred == 1:  # Enter position
            in_position = True
            entry_idx = i
            collected_rates = [actual_rate] if actual_rate > 0 else []
            entry_date = dt

        elif in_position and pred == 0:  # Exit position
            # Calculate P&L
            gross_pnl = sum(collected_rates)
            net_pnl = gross_pnl - TOTAL_COST

            trades.append({
                'entry_date': entry_date,
                'exit_date': dt,
                'periods_held': len(collected_rates),
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'fees_paid': TOTAL_COST,
                'profitable': net_pnl > 0
            })

            in_position = False
            collected_rates = []

        elif in_position and pred == 1:  # Stay in position
            if actual_rate > 0:
                collected_rates.append(actual_rate)

    return pd.DataFrame(trades)

# Run backtest
print("\n" + "-"*50)
print("STRATEGY BACKTEST (Using ML Predictions)")
print("-"*50)

trades_df = backtest_strategy(results_df, df, TOTAL_COST)

if len(trades_df) > 0:
    total_trades = len(trades_df)
    winning_trades = trades_df['profitable'].sum()
    total_gross_pnl = trades_df['gross_pnl'].sum()
    total_net_pnl = trades_df['net_pnl'].sum()
    total_fees = trades_df['fees_paid'].sum()
    avg_holding = trades_df['periods_held'].mean()

    print(f"""
Backtest Results:
  - Total Trades: {total_trades}
  - Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.1f}%)
  - Losing Trades: {total_trades - winning_trades} ({(total_trades-winning_trades)/total_trades*100:.1f}%)

  - Gross P&L: {total_gross_pnl*100:.4f}%
  - Total Fees Paid: {total_fees*100:.4f}%
  - Net P&L: {total_net_pnl*100:.4f}%

  - Avg P&L per Trade: {trades_df['net_pnl'].mean()*100:.4f}%
  - Avg Holding Period: {avg_holding:.1f} periods ({avg_holding/3:.1f} days)
""")

    # Show trade details
    print("\nTrade Details (first 20):")
    print(trades_df.head(20).to_string(index=False))
else:
    print("No trades executed in backtest period.")

#%% Profitability Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Streak Length Distribution
ax1 = axes[0, 0]
ax1.bar(streak_summary['length'], streak_summary['count'], color='steelblue', alpha=0.7)
ax1.set_xlabel('Streak Length (periods)')
ax1.set_ylabel('Count')
ax1.set_title(f'Distribution of Consecutive Positive Funding Streaks - {SYMBOL}')
ax1.axvline(x=breakeven_avg, color='red', linestyle='--', label=f'Breakeven ({breakeven_avg:.1f})')
ax1.legend()

# 2. Net Return by Streak Length
ax2 = axes[0, 1]
colors = ['green' if x else 'red' for x in streak_summary['profitable']]
ax2.bar(streak_summary['length'], streak_summary['net_return'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Streak Length (periods)')
ax2.set_ylabel('Net Return (%)')
ax2.set_title(f'Net Return by Streak Length (after {TOTAL_COST*100:.2f}% fees)')

# 3. Cumulative P&L from Backtest
ax3 = axes[1, 0]
if len(trades_df) > 0:
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum() * 100
    ax3.plot(range(len(trades_df)), trades_df['cumulative_pnl'], color='blue', linewidth=2)
    ax3.fill_between(range(len(trades_df)), 0, trades_df['cumulative_pnl'],
                     where=trades_df['cumulative_pnl'] >= 0, alpha=0.3, color='green')
    ax3.fill_between(range(len(trades_df)), 0, trades_df['cumulative_pnl'],
                     where=trades_df['cumulative_pnl'] < 0, alpha=0.3, color='red')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Trade Number')
ax3.set_ylabel('Cumulative P&L (%)')
ax3.set_title('Cumulative P&L from ML-Based Strategy')

# 4. Breakeven Sensitivity Analysis
ax4 = axes[1, 1]
fee_levels = np.arange(0.001, 0.005, 0.0005)  # 0.1% to 0.5%
breakeven_periods = [calculate_breakeven(avg_positive_rate, f) for f in fee_levels]
ax4.plot(fee_levels * 100, breakeven_periods, color='purple', linewidth=2, marker='o')
ax4.axhline(y=21, color='red', linestyle='--', alpha=0.5, label='1 week (21 periods)')
ax4.axhline(y=63, color='orange', linestyle='--', alpha=0.5, label='3 weeks (63 periods)')
ax4.set_xlabel('Total Round-Trip Cost (%)')
ax4.set_ylabel('Breakeven Periods')
ax4.set_title('Breakeven Sensitivity to Fees')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(f'Profitability Analysis - {SYMBOL}', y=1.02, fontsize=14)
plt.show()

#%% Trading Strategy Summary
print("\n" + "="*70)
print("TRADING STRATEGY SUMMARY")
print("="*70)

# Historical opportunity analysis
profitable_streaks = streak_df[streak_df['length'] >= breakeven_avg] if not pd.isna(min_profitable_streak) else pd.DataFrame()
total_streaks = len(streak_df)
profitable_opportunities = len(profitable_streaks)

print(f"""
SYMBOL: {SYMBOL}

COST STRUCTURE:
  - Entry Fee: {ENTRY_FEE*100:.2f}%
  - Exit Fee: {EXIT_FEE*100:.2f}%
  - Slippage: {SLIPPAGE*100:.3f}% x 2
  - Total Round-Trip: {TOTAL_COST*100:.2f}%

BREAKEVEN REQUIREMENTS:
  - Periods needed: {breakeven_avg:.1f} ({breakeven_avg/3:.1f} days)
  - Min profitable streak: {min_profitable_streak if not pd.isna(min_profitable_streak) else 'N/A'} periods

HISTORICAL OPPORTUNITIES:
  - Total positive streaks: {total_streaks}
  - Profitable streaks (>= breakeven): {profitable_opportunities} ({profitable_opportunities/total_streaks*100:.1f}% of streaks)
""")

if len(trades_df) > 0:
    print(f"""
ML STRATEGY PERFORMANCE:
  - Total trades: {len(trades_df)}
  - Win rate: {trades_df['profitable'].mean()*100:.1f}%
  - Net P&L: {total_net_pnl*100:.4f}%
  - Avg trade P&L: {trades_df['net_pnl'].mean()*100:.4f}%
""")

print("""
RECOMMENDATION:
""")
if not pd.isna(min_profitable_streak) and min_profitable_streak <= 21:
    print(f"  ✓ Strategy may be viable - breakeven achievable within {breakeven_avg/3:.1f} days")
else:
    print(f"  ✗ Strategy challenging - need {breakeven_avg/3:.1f}+ days of consecutive positive rates")

if len(trades_df) > 0 and total_net_pnl > 0:
    print(f"  ✓ ML strategy shows positive returns in backtest: {total_net_pnl*100:.4f}%")
elif len(trades_df) > 0:
    print(f"  ✗ ML strategy shows negative returns: {total_net_pnl*100:.4f}% - consider longer holding periods")

#%% ============================================================================
# BUY & HOLD FUNDING RATE ANALYSIS
# ============================================================================

#%% Buy & Hold Analysis
print("\n" + "="*70)
print("BUY & HOLD FUNDING RATE ANALYSIS")
print("="*70)
print("Strategy: Hold short perpetual position continuously (Long Spot + Short Perp)")

# Calculate total funding collected
total_funding = df['fundingRate'].sum()
positive_funding = df[df['fundingRate'] > 0]['fundingRate'].sum()
negative_funding = df[df['fundingRate'] < 0]['fundingRate'].sum()

# Time period
start_date = df['fundingDateTime'].min()
end_date = df['fundingDateTime'].max()
days_held = (end_date - start_date).days
years_held = days_held / 365

# Annualized rate
annualized_rate = (total_funding / days_held) * 365 if days_held > 0 else 0

# Count positive vs negative periods
n_positive = (df['fundingRate'] > 0).sum()
n_negative = (df['fundingRate'] < 0).sum()
n_zero = (df['fundingRate'] == 0).sum()

print(f"""
HOLDING PERIOD:
  - Start: {start_date.date()}
  - End: {end_date.date()}
  - Duration: {days_held} days ({years_held:.2f} years)
  - Total Periods: {len(df):,}

FUNDING RATE SUMMARY:
  - Total Funding Collected: {total_funding*100:.4f}%
  - From Positive Rates: +{positive_funding*100:.4f}%
  - From Negative Rates: {negative_funding*100:.4f}%

  - Positive Periods: {n_positive:,} ({n_positive/len(df)*100:.1f}%)
  - Negative Periods: {n_negative:,} ({n_negative/len(df)*100:.1f}%)
  - Zero Periods: {n_zero:,} ({n_zero/len(df)*100:.1f}%)

ANNUALIZED RETURNS:
  - Annualized Funding Rate: {annualized_rate*100:.2f}%
  - Average Daily Rate: {(total_funding/days_held)*100:.4f}%
  - Average Per Period: {(total_funding/len(df))*100:.4f}%
""")

#%% Monthly Funding Breakdown
df_monthly = df.copy()
df_monthly['month'] = df_monthly['fundingDateTime'].dt.to_period('M')
monthly_funding = df_monthly.groupby('month')['fundingRate'].agg(['sum', 'mean', 'count']).reset_index()
monthly_funding.columns = ['month', 'total_rate', 'avg_rate', 'periods']
monthly_funding['total_pct'] = monthly_funding['total_rate'] * 100
monthly_funding['annualized'] = monthly_funding['avg_rate'] * 3 * 365 * 100  # 3 periods/day * 365 days

print("\n" + "-"*50)
print("MONTHLY FUNDING BREAKDOWN")
print("-"*50)
print("\nLast 24 Months:")
print(monthly_funding.tail(24)[['month', 'total_pct', 'annualized', 'periods']].to_string(index=False))

#%% Yearly Funding Summary
df_yearly = df.copy()
df_yearly['year'] = df_yearly['fundingDateTime'].dt.year
yearly_funding = df_yearly.groupby('year')['fundingRate'].agg(['sum', 'mean', 'count']).reset_index()
yearly_funding.columns = ['year', 'total_rate', 'avg_rate', 'periods']
yearly_funding['total_pct'] = yearly_funding['total_rate'] * 100
yearly_funding['annualized'] = yearly_funding['avg_rate'] * 3 * 365 * 100

print("\n" + "-"*50)
print("YEARLY FUNDING SUMMARY")
print("-"*50)
print(yearly_funding.to_string(index=False))

#%% Buy & Hold Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cumulative Funding Over Time
ax1 = axes[0, 0]
df['cumulative_funding'] = df['fundingRate'].cumsum() * 100
ax1.plot(df['fundingDateTime'], df['cumulative_funding'], color='blue', linewidth=1)
ax1.fill_between(df['fundingDateTime'], 0, df['cumulative_funding'],
                 where=df['cumulative_funding'] >= 0, alpha=0.3, color='green')
ax1.fill_between(df['fundingDateTime'], 0, df['cumulative_funding'],
                 where=df['cumulative_funding'] < 0, alpha=0.3, color='red')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Funding Rate (%)')
ax1.set_title(f'Buy & Hold Cumulative Funding - {SYMBOL}')

# 2. Monthly Funding Rate
ax2 = axes[0, 1]
colors = ['green' if x > 0 else 'red' for x in monthly_funding['total_pct']]
ax2.bar(range(len(monthly_funding)), monthly_funding['total_pct'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Month Index')
ax2.set_ylabel('Monthly Funding Rate (%)')
ax2.set_title('Monthly Funding Rate')

# 3. Yearly Funding Comparison
ax3 = axes[1, 0]
colors = ['green' if x > 0 else 'red' for x in yearly_funding['total_pct']]
bars = ax3.bar(yearly_funding['year'].astype(str), yearly_funding['total_pct'], color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Year')
ax3.set_ylabel('Total Funding Rate (%)')
ax3.set_title('Yearly Funding Rate')
for bar, val in zip(bars, yearly_funding['total_pct']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

# 4. Funding Rate Distribution
ax4 = axes[1, 1]
ax4.hist(df['fundingRate'] * 100, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax4.axvline(x=df['fundingRate'].mean() * 100, color='green', linestyle='-', linewidth=2,
            label=f'Mean: {df["fundingRate"].mean()*100:.4f}%')
ax4.set_xlabel('Funding Rate (%)')
ax4.set_ylabel('Frequency')
ax4.set_title('Funding Rate Distribution')
ax4.legend()

plt.tight_layout()
plt.suptitle(f'Buy & Hold Funding Analysis - {SYMBOL}', y=1.02, fontsize=14)
plt.show()

#%% Buy & Hold Summary
print("\n" + "="*70)
print("BUY & HOLD SUMMARY")
print("="*70)

# Net after fees (one-time entry/exit)
net_after_fees = total_funding - TOTAL_COST
net_annualized = (net_after_fees / days_held) * 365 if days_held > 0 else 0

print(f"""
SYMBOL: {SYMBOL}
PERIOD: {start_date.date()} to {end_date.date()} ({days_held} days)

GROSS RETURNS (before fees):
  - Total Funding: {total_funding*100:.4f}%
  - Annualized: {annualized_rate*100:.2f}%

NET RETURNS (after {TOTAL_COST*100:.2f}% round-trip fees):
  - Net Total: {net_after_fees*100:.4f}%
  - Net Annualized: {net_annualized*100:.2f}%

COMPARISON:
  - Buy & Hold Annualized: {annualized_rate*100:.2f}%
  - ML Strategy Return: {total_net_pnl*100:.4f}% (over backtest period)
""")

if annualized_rate > 0.05:  # > 5% annualized
    print(f"  ✓ Buy & Hold shows attractive {annualized_rate*100:.1f}% annualized return")
else:
    print(f"  ⚠ Buy & Hold shows modest {annualized_rate*100:.1f}% annualized return")

#%% ============================================================================
# STREAK PREDICTION MODEL
# ============================================================================

#%% Profitable Streak Target Creation
print("\n" + "="*70)
print("PROFITABLE STREAK PREDICTION")
print("="*70)

# Use breakeven as minimum profitable streak length
MIN_PROFITABLE_STREAK = int(np.ceil(breakeven_avg))
print(f"Minimum profitable streak length: {MIN_PROFITABLE_STREAK} periods ({MIN_PROFITABLE_STREAK/3:.1f} days)")

def create_streak_target(funding_rates, min_streak_length):
    """
    Create target: Is this period the START of a profitable streak?

    A profitable streak start means:
    1. Current rate is positive
    2. Previous rate was negative (or first period)
    3. The streak starting here has length >= min_streak_length
    """
    n = len(funding_rates)
    target = np.zeros(n, dtype=int)

    i = 0
    while i < n:
        # Check if this could be a streak start (positive after non-positive)
        if funding_rates[i] > 0 and (i == 0 or funding_rates[i-1] <= 0):
            # Count streak length
            streak_len = 0
            j = i
            while j < n and funding_rates[j] > 0:
                streak_len += 1
                j += 1

            # Mark as profitable streak start if long enough
            if streak_len >= min_streak_length:
                target[i] = 1

            i = j  # Skip to end of streak
        else:
            i += 1

    return target

# Create streak target
df['streak_start_target'] = create_streak_target(df['fundingRate'].values, MIN_PROFITABLE_STREAK)

# Check class distribution
n_positive_targets = df['streak_start_target'].sum()
print(f"\nTarget Distribution:")
print(f"  - Profitable streak starts: {n_positive_targets} ({n_positive_targets/len(df)*100:.2f}%)")
print(f"  - Non-starts: {len(df) - n_positive_targets} ({(len(df)-n_positive_targets)/len(df)*100:.2f}%)")

#%% Additional Streak Features
def create_streak_features(df):
    """Create additional features for streak prediction."""
    df = df.copy()

    # Previous streak information
    prev_streak_len = []
    time_since_streak = []
    current_streak = 0
    last_streak_len = 0
    periods_since = 0

    for i, rate in enumerate(df['fundingRate']):
        if rate > 0:
            current_streak += 1
            periods_since = 0
        else:
            if current_streak > 0:
                last_streak_len = current_streak
            current_streak = 0
            periods_since += 1

        prev_streak_len.append(last_streak_len)
        time_since_streak.append(periods_since)

    df['prev_streak_length'] = prev_streak_len
    df['time_since_streak'] = time_since_streak

    # Rolling positive ratio (what % of last N periods were positive?)
    df['rolling_pos_ratio_12'] = (df['fundingRate'] > 0).rolling(12).mean()
    df['rolling_pos_ratio_24'] = (df['fundingRate'] > 0).rolling(24).mean()

    # Rate trend (is funding rate trending up?)
    df['rate_trend_6'] = df['fundingRate'].rolling(6).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else 0
    )

    # Volatility of recent rates
    df['rate_volatility_12'] = df['fundingRate'].rolling(12).std()

    # Average positive rate in last N periods (when positive)
    def avg_positive_rate(x):
        pos = x[x > 0]
        return pos.mean() if len(pos) > 0 else 0

    df['avg_pos_rate_24'] = df['fundingRate'].rolling(24).apply(avg_positive_rate)

    return df

df_streak = create_streak_features(df)
print(f"\nAdded {5} streak-specific features")

#%% Prepare Streak Model Data
# Combine with existing features (from df_ml which has lag features etc)
# We need to align the indices

# Use original feature columns plus new streak features
streak_feature_cols = feature_cols + ['prev_streak_length', 'time_since_streak',
                                       'rolling_pos_ratio_12', 'rolling_pos_ratio_24',
                                       'rate_trend_6', 'rate_volatility_12', 'avg_pos_rate_24']

# Merge streak features into df_ml
for col in ['prev_streak_length', 'time_since_streak', 'rolling_pos_ratio_12',
            'rolling_pos_ratio_24', 'rate_trend_6', 'rate_volatility_12',
            'avg_pos_rate_24', 'streak_start_target']:
    df_ml[col] = df_streak[col].values[:len(df_ml)] if col in df_streak.columns else 0

# Drop NaN from new features
df_streak_ml = df_ml.dropna()
print(f"Records for streak model: {len(df_streak_ml):,}")

# Prepare X and y
streak_feature_cols_available = [c for c in streak_feature_cols if c in df_streak_ml.columns]
X_streak = df_streak_ml[streak_feature_cols_available].values
y_streak = df_streak_ml['streak_start_target'].values

print(f"Features for streak model: {len(streak_feature_cols_available)}")
print(f"Positive class ratio: {y_streak.mean()*100:.2f}%")

#%% Train Streak Start Classifier
print("\n" + "-"*50)
print("TRAINING STREAK START CLASSIFIER")
print("-"*50)

# Time-based split
split_idx_streak = int(len(X_streak) * 0.8)
X_train_streak = X_streak[:split_idx_streak]
X_test_streak = X_streak[split_idx_streak:]
y_train_streak = y_streak[:split_idx_streak]
y_test_streak = y_streak[split_idx_streak:]

print(f"Train: {len(X_train_streak)}, Test: {len(X_test_streak)}")
print(f"Train positive ratio: {y_train_streak.mean()*100:.2f}%")
print(f"Test positive ratio: {y_test_streak.mean()*100:.2f}%")

# Scale features
scaler_streak = StandardScaler()
X_train_streak_scaled = scaler_streak.fit_transform(X_train_streak)
X_test_streak_scaled = scaler_streak.transform(X_test_streak)

# Train models with class_weight='balanced' to handle imbalance
streak_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                                               class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                             class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

if HAS_XGBOOST:
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (len(y_train_streak) - y_train_streak.sum()) / max(y_train_streak.sum(), 1)
    streak_models['XGBoost'] = XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric='logloss'
    )

streak_results = {}

for name, model in streak_models.items():
    print(f"\nTraining {name}...")

    if name in ['Logistic Regression']:
        model.fit(X_train_streak_scaled, y_train_streak)
        y_pred = model.predict(X_test_streak_scaled)
        y_prob = model.predict_proba(X_test_streak_scaled)[:, 1]
    else:
        model.fit(X_train_streak, y_train_streak)
        y_pred = model.predict(X_test_streak)
        y_prob = model.predict_proba(X_test_streak)[:, 1]

    # Metrics focused on precision (we want high confidence signals)
    acc = accuracy_score(y_test_streak, y_pred)
    prec = precision_score(y_test_streak, y_pred, zero_division=0)
    rec = recall_score(y_test_streak, y_pred, zero_division=0)
    f1 = f1_score(y_test_streak, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_test_streak, y_prob)
    except:
        roc = 0.5

    streak_results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f} (if we predict 'start', how often correct?)")
    print(f"  Recall: {rec:.4f} (what % of actual starts do we catch?)")
    print(f"  F1: {f1:.4f}")

#%% Streak Model Comparison
print("\n" + "-"*50)
print("STREAK MODEL COMPARISON")
print("-"*50)

streak_summary = pd.DataFrame({
    name: {
        'Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1': f"{r['f1']:.4f}",
        'ROC-AUC': f"{r['roc_auc']:.4f}"
    }
    for name, r in streak_results.items()
}).T

print(streak_summary.to_string())

# Best model by F1 (balance precision and recall)
best_streak_model_name = max(streak_results, key=lambda x: streak_results[x]['f1'])
print(f"\nBest Streak Model (by F1): {best_streak_model_name}")

#%% Threshold Tuning for High Precision
print("\n" + "-"*50)
print("THRESHOLD TUNING FOR HIGH PRECISION")
print("-"*50)

best_streak_result = streak_results[best_streak_model_name]
y_prob_streak = best_streak_result['y_prob']

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []

for thresh in thresholds:
    y_pred_thresh = (y_prob_streak >= thresh).astype(int)
    n_predictions = y_pred_thresh.sum()

    if n_predictions > 0:
        prec = precision_score(y_test_streak, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test_streak, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test_streak, y_pred_thresh, zero_division=0)
    else:
        prec, rec, f1 = 0, 0, 0

    threshold_results.append({
        'threshold': thresh,
        'predictions': n_predictions,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nThreshold Analysis:")
print(threshold_df.to_string(index=False))

# Find threshold with precision >= 50%
high_prec_thresholds = threshold_df[threshold_df['precision'] >= 0.5]
if len(high_prec_thresholds) > 0:
    best_threshold = high_prec_thresholds.iloc[0]['threshold']
    print(f"\nRecommended threshold for >=50% precision: {best_threshold}")
else:
    best_threshold = 0.5
    print(f"\nUsing default threshold: {best_threshold}")

#%% Streak-Based Backtest
print("\n" + "-"*50)
print("STREAK-BASED BACKTEST")
print("-"*50)

def backtest_streak_strategy(y_test, y_prob, actual_rates, dates, threshold, min_hold_periods):
    """
    Backtest using streak predictions.

    Entry: When model predicts streak start (prob >= threshold)
    Exit: After min_hold_periods OR when rate turns negative
    """
    trades = []
    in_position = False
    entry_idx = None
    collected_rates = []

    for i in range(len(y_test)):
        pred_start = y_prob[i] >= threshold
        actual_rate = actual_rates[i] if i < len(actual_rates) else 0

        if not in_position and pred_start:
            # Enter position
            in_position = True
            entry_idx = i
            collected_rates = []
            entry_date = dates[i]
            hold_count = 0

        if in_position:
            # Collect funding
            if actual_rate > 0:
                collected_rates.append(actual_rate)
            hold_count = i - entry_idx + 1

            # Exit conditions: held enough periods OR rate turned negative
            should_exit = (hold_count >= min_hold_periods) or (actual_rate <= 0 and hold_count > 1)

            if should_exit or i == len(y_test) - 1:
                gross_pnl = sum(collected_rates)
                net_pnl = gross_pnl - TOTAL_COST

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': dates[i],
                    'periods_held': len(collected_rates),
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'profitable': net_pnl > 0
                })

                in_position = False
                collected_rates = []

    return pd.DataFrame(trades)

# Get actual rates for test period
test_dates = df_streak_ml['fundingDateTime'].iloc[split_idx_streak:].values
test_rates = df_streak_ml['fundingRate'].iloc[split_idx_streak:].values

# Run streak-based backtest
streak_trades = backtest_streak_strategy(
    y_test_streak, y_prob_streak, test_rates, test_dates,
    threshold=best_threshold, min_hold_periods=MIN_PROFITABLE_STREAK
)

if len(streak_trades) > 0:
    print(f"\nStreak-Based Strategy Results (threshold={best_threshold}):")
    print(f"  - Total Trades: {len(streak_trades)}")
    print(f"  - Winning Trades: {streak_trades['profitable'].sum()} ({streak_trades['profitable'].mean()*100:.1f}%)")
    print(f"  - Total Net P&L: {streak_trades['net_pnl'].sum()*100:.4f}%")
    print(f"  - Avg P&L per Trade: {streak_trades['net_pnl'].mean()*100:.4f}%")
    print(f"  - Avg Holding Period: {streak_trades['periods_held'].mean():.1f} periods")

    print("\nTrade Details:")
    print(streak_trades.head(20).to_string(index=False))
else:
    print("No trades generated with streak strategy.")

#%% Comparison: Single vs Streak Prediction
print("\n" + "="*70)
print("STRATEGY COMPARISON")
print("="*70)

print(f"""
                        | Single Period | Streak-Based |
------------------------|---------------|--------------|""")

if len(trades_df) > 0 and len(streak_trades) > 0:
    print(f"Total Trades            | {len(trades_df):13} | {len(streak_trades):12} |")
    print(f"Win Rate                | {trades_df['profitable'].mean()*100:12.1f}% | {streak_trades['profitable'].mean()*100:11.1f}% |")
    print(f"Total Net P&L           | {trades_df['net_pnl'].sum()*100:12.4f}% | {streak_trades['net_pnl'].sum()*100:11.4f}% |")
    print(f"Avg P&L/Trade           | {trades_df['net_pnl'].mean()*100:12.4f}% | {streak_trades['net_pnl'].mean()*100:11.4f}% |")
    print(f"Avg Holding (periods)   | {trades_df['periods_held'].mean():13.1f} | {streak_trades['periods_held'].mean():12.1f} |")
elif len(trades_df) > 0:
    print(f"Total Trades            | {len(trades_df):13} | {'N/A':>12} |")
    print(f"Win Rate                | {trades_df['profitable'].mean()*100:12.1f}% | {'N/A':>12} |")
    print(f"Total Net P&L           | {trades_df['net_pnl'].sum()*100:12.4f}% | {'N/A':>12} |")

print(f"""
RECOMMENDATION:
""")

if len(streak_trades) > 0 and len(trades_df) > 0:
    if streak_trades['net_pnl'].sum() > trades_df['net_pnl'].sum():
        print("  ✓ Streak-based strategy outperforms single-period prediction")
    else:
        print("  ⚠ Single-period strategy performs better in this backtest")

    if streak_trades['profitable'].mean() > 0.5:
        print(f"  ✓ Streak strategy has {streak_trades['profitable'].mean()*100:.0f}% win rate")
    else:
        print(f"  ⚠ Streak strategy win rate is {streak_trades['profitable'].mean()*100:.0f}% - consider higher threshold")

# %%
