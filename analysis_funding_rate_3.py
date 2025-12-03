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
SYMBOL = 'SOLUSDT'  # Change to any symbol: 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', etc.
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

# %%
