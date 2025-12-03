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
TEST_SIZE = 0.2     # 20% for testing
RANDOM_STATE = 42

# Data directory
DATA_DIR = Path(__file__).parent / "raw_funding_rate"
if not DATA_DIR.exists():
    DATA_DIR = Path("raw_funding_rate")

print(f"Selected Symbol: {SYMBOL}")
print(f"Test Size: {TEST_SIZE * 100:.0f}%")

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

#%% Train/Test Split (Time-based)
# IMPORTANT: Use time-based split, NOT random split (to avoid data leakage)
split_idx = int(len(X) * (1 - TEST_SIZE))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")
print(f"Train period: {df_ml['fundingDateTime'].iloc[0]} to {df_ml['fundingDateTime'].iloc[split_idx-1]}")
print(f"Test period: {df_ml['fundingDateTime'].iloc[split_idx]} to {df_ml['fundingDateTime'].iloc[-1]}")

#%% Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Define Models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )

print(f"\nModels to train: {list(models.keys())}")

#%% Train and Evaluate Models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled data for SVM and Logistic Regression
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'roc_auc': roc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC:  {roc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

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

#%% Feature Importance (Random Forest)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    importance = rf_model.feature_importances_

    feat_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
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
if HAS_XGBOOST and 'XGBoost' in results:
    xgb_model = results['XGBoost']['model']
    importance_xgb = xgb_model.feature_importances_

    feat_importance_xgb = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance_xgb
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

#%% Cross-Validation with TimeSeriesSplit
print("\n" + "="*70)
print("TIME SERIES CROSS-VALIDATION")
print("="*70)

tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}

for name, model in models.items():
    # Use appropriate data (scaled or not)
    if name in ['SVM', 'Logistic Regression']:
        X_cv = scaler.fit_transform(X)
    else:
        X_cv = X

    # Re-instantiate model for fresh CV
    if name == 'Logistic Regression':
        cv_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif name == 'Random Forest':
        cv_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == 'Gradient Boosting':
        cv_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    elif name == 'SVM':
        cv_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    elif name == 'XGBoost' and HAS_XGBOOST:
        cv_model = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                  use_label_encoder=False, eval_metric='logloss')
    else:
        continue

    scores = cross_val_score(cv_model, X_cv, y, cv=tscv, scoring='accuracy')
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }

    print(f"\n{name}:")
    print(f"  CV Scores: {scores}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

#%% Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Test set performance
metrics_df = pd.DataFrame({
    name: [r['accuracy'], r['roc_auc'], r['f1']]
    for name, r in results.items()
}, index=['Accuracy', 'ROC-AUC', 'F1 Score'])

metrics_df.T.plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_title(f'Test Set Performance - {SYMBOL}')
axes[0].set_ylabel('Score')
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1)

# Cross-validation performance
cv_means = [cv_results[name]['mean'] for name in cv_results]
cv_stds = [cv_results[name]['std'] for name in cv_results]

axes[1].bar(cv_results.keys(), cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
axes[1].set_title(f'Cross-Validation Accuracy - {SYMBOL}')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%% Prediction Timeline
best_result = results[best_model_name]
test_dates = df_ml['fundingDateTime'].iloc[split_idx:].values

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Actual vs Predicted
axes[0].plot(test_dates, y_test, label='Actual', alpha=0.7)
axes[0].plot(test_dates, best_result['y_pred'], label='Predicted', alpha=0.7, linestyle='--')
axes[0].set_ylabel('Direction (0=Neg, 1=Pos)')
axes[0].set_title(f'{best_model_name} Predictions vs Actual - {SYMBOL}')
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
print(classification_report(y_test, best_result['y_pred'],
                            target_names=['Negative', 'Positive']))

#%% Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Symbol: {SYMBOL}
Total Records: {len(df_ml):,}
Features Used: {len(feature_cols)}
Train/Test Split: {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%

Best Model: {best_model_name}
  - Test Accuracy: {results[best_model_name]['accuracy']:.4f}
  - Test ROC-AUC: {results[best_model_name]['roc_auc']:.4f}
  - Test F1 Score: {results[best_model_name]['f1']:.4f}

Class Distribution:
  - Positive rates: {df_ml['target'].sum():,} ({df_ml['target'].mean()*100:.1f}%)
  - Negative rates: {len(df_ml) - df_ml['target'].sum():,} ({(1-df_ml['target'].mean())*100:.1f}%)
""")

print("\nTo change the symbol, modify SYMBOL at the top of this script.")
print("Available symbols can be found in the raw_funding_rate folder.")

#%% Final Results DataFrame
# Create DataFrame with actual vs predicted directions
best_result = results[best_model_name]

results_df = pd.DataFrame({
    'datetime': df_ml['fundingDateTime'].iloc[split_idx:].values,
    'current_rate': df_ml['fundingRate'].iloc[split_idx:].values,
    'next_rate': df_ml['fundingRate'].shift(-1).iloc[split_idx:].values,  # The rate we're predicting
    'actual_direction': y_test,  # Direction of NEXT rate
    'predicted_direction': best_result['y_pred'],
    'prediction_prob': best_result['y_prob'],
    'correct': (y_test == best_result['y_pred']).astype(int)
})

# Map direction to readable labels
results_df['actual_label'] = results_df['actual_direction'].map({1: 'Positive', 0: 'Negative'})
results_df['predicted_label'] = results_df['predicted_direction'].map({1: 'Positive', 0: 'Negative'})

print(f"\n{'='*70}")
print(f"PREDICTION RESULTS - {best_model_name} on {SYMBOL}")
print(f"{'='*70}")
print(f"\nTotal predictions: {len(results_df)}")
print(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.1f}%)")

print("\nFirst 20 predictions:")
print(results_df[['datetime', 'current_rate', 'next_rate', 'actual_label', 'predicted_label', 'prediction_prob', 'correct']].head(20).to_string(index=False))

print("\nLast 20 predictions:")
print(results_df[['datetime', 'current_rate', 'next_rate', 'actual_label', 'predicted_label', 'prediction_prob', 'correct']].tail(20).to_string(index=False))

# Export to CSV (optional)
results_df.to_csv(f'{SYMBOL}_predictions.csv', index=False)

# %%
