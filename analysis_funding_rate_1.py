#%% [markdown]
# Funding Rate Analysis
# This script analyzes funding rate data from Binance futures

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

#%% Configuration
DATA_DIR = Path(__file__).parent / "raw_funding_rate"
# For Jupyter notebook compatibility
if not DATA_DIR.exists():
    DATA_DIR = Path("raw_funding_rate")

#%% Load all funding rate files dynamically
def load_all_funding_rates(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files from the raw_funding_rate directory."""
    all_data = {}
    csv_files = list(data_dir.glob("*_funding_rate_*.csv"))

    print(f"Found {len(csv_files)} funding rate files:")
    for file in csv_files:
        symbol = file.name.split("_")[0]
        print(f"  - {symbol}")

        df = pd.read_csv(file)
        df['fundingDateTime'] = pd.to_datetime(df['fundingDateTime'])
        df['formattedFundingDateTime'] = pd.to_datetime(df['formattedFundingDateTime'])
        df = df.sort_values('fundingDateTime').reset_index(drop=True)
        all_data[symbol] = df

    return all_data

funding_data = load_all_funding_rates(DATA_DIR)

#%% Create combined dataframe
def create_combined_df(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all funding rate data into a single DataFrame."""
    dfs = []
    for symbol, df in data.items():
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined

combined_df = create_combined_df(funding_data)
print(f"Total records: {len(combined_df):,}")
print(f"Symbols: {combined_df['symbol'].nunique()}")
print(f"Date range: {combined_df['fundingDateTime'].min()} to {combined_df['fundingDateTime'].max()}")

#%% Summary statistics per symbol
def get_funding_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate summary statistics for each symbol."""
    stats = []
    for symbol, df in data.items():
        stats.append({
            'symbol': symbol,
            'count': len(df),
            'mean_rate': df['fundingRate'].mean(),
            'median_rate': df['fundingRate'].median(),
            'std_rate': df['fundingRate'].std(),
            'min_rate': df['fundingRate'].min(),
            'max_rate': df['fundingRate'].max(),
            'positive_pct': (df['fundingRate'] > 0).mean() * 100,
            'annualized_rate': df['fundingRate'].mean() * 3 * 365 * 100,  # 3 funding periods per day
            'start_date': df['fundingDateTime'].min(),
            'end_date': df['fundingDateTime'].max(),
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('annualized_rate', ascending=False)
    return stats_df

stats_df = get_funding_stats(funding_data)
print("\nFunding Rate Statistics:")
print(stats_df.to_string(index=False))

#%% Plot funding rates over time
def plot_funding_rates(data: dict[str, pd.DataFrame], symbols: list[str] = None):
    """Plot funding rates over time for selected symbols."""
    if symbols is None:
        symbols = list(data.keys())[:5]  # Default to first 5 symbols

    fig, ax = plt.subplots(figsize=(14, 7))

    for symbol in symbols:
        if symbol in data:
            df = data[symbol]
            ax.plot(df['fundingDateTime'], df['fundingRate'] * 100, label=symbol, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Funding Rate (%)')
    ax.set_title('Funding Rates Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Plot major coins
major_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
available_major = [c for c in major_coins if c in funding_data]
plot_funding_rates(funding_data, available_major)

#%% Calculate rolling average funding rate
def calc_rolling_funding(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calculate rolling average funding rate (in number of 8-hour periods)."""
    df = df.copy()
    df['rolling_rate'] = df['fundingRate'].rolling(window=window * 3).mean()  # 3 periods per day
    df['cumulative_rate'] = df['fundingRate'].cumsum()
    return df

# Example: Calculate rolling funding for BTC
if 'BTCUSDT' in funding_data:
    btc_rolling = calc_rolling_funding(funding_data['BTCUSDT'], window=30)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(btc_rolling['fundingDateTime'], btc_rolling['rolling_rate'] * 100)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('30-Day Rolling Avg Rate (%)')
    axes[0].set_title('BTCUSDT 30-Day Rolling Average Funding Rate')

    axes[1].plot(btc_rolling['fundingDateTime'], btc_rolling['cumulative_rate'] * 100)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Rate (%)')
    axes[1].set_title('BTCUSDT Cumulative Funding Rate')

    plt.tight_layout()
    plt.show()

#%% Funding rate distribution
def plot_funding_distribution(data: dict[str, pd.DataFrame], symbols: list[str] = None):
    """Plot funding rate distribution for selected symbols."""
    if symbols is None:
        symbols = list(data.keys())[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, symbol in enumerate(symbols[:4]):
        if symbol in data:
            df = data[symbol]
            axes[i].hist(df['fundingRate'] * 100, bins=50, edgecolor='black', alpha=0.7)
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[i].axvline(x=df['fundingRate'].mean() * 100, color='green', linestyle='-',
                          label=f'Mean: {df["fundingRate"].mean() * 100:.4f}%')
            axes[i].set_xlabel('Funding Rate (%)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{symbol} Funding Rate Distribution')
            axes[i].legend()

    plt.tight_layout()
    plt.show()

plot_funding_distribution(funding_data, available_major[:4])

#%% Extreme funding rate events
def find_extreme_events(data: dict[str, pd.DataFrame], threshold: float = 0.01) -> pd.DataFrame:
    """Find extreme funding rate events (above threshold in absolute value)."""
    extreme_events = []

    for symbol, df in data.items():
        extreme = df[abs(df['fundingRate']) > threshold].copy()
        if len(extreme) > 0:
            extreme_events.append(extreme)

    if extreme_events:
        result = pd.concat(extreme_events, ignore_index=True)
        result = result.sort_values('fundingDateTime', ascending=False)
        return result
    return pd.DataFrame()

extreme_df = find_extreme_events(funding_data, threshold=0.005)  # 0.5% threshold
print(f"\nExtreme Funding Events (|rate| > 0.5%):")
print(f"Total events: {len(extreme_df)}")
if len(extreme_df) > 0:
    print(extreme_df[['symbol', 'fundingRate', 'fundingDateTime']].head(20).to_string(index=False))

#%% Correlation between funding rates
def calc_funding_correlation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate correlation matrix between funding rates of different symbols."""
    # Pivot to wide format
    all_rates = []
    for symbol, df in data.items():
        temp = df[['fundingDateTime', 'fundingRate']].copy()
        temp = temp.rename(columns={'fundingRate': symbol})
        temp = temp.set_index('fundingDateTime')
        all_rates.append(temp)

    if len(all_rates) < 2:
        return pd.DataFrame()

    merged = all_rates[0]
    for df in all_rates[1:]:
        merged = merged.join(df, how='outer')

    return merged.corr()

corr_matrix = calc_funding_correlation(funding_data)
if len(corr_matrix) > 0:
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title('Funding Rate Correlation Matrix')
    plt.tight_layout()
    plt.show()

#%% Monthly average funding rates
def calc_monthly_funding(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate monthly average funding rates."""
    monthly_data = []

    for symbol, df in data.items():
        df = df.copy()
        df['month'] = df['fundingDateTime'].dt.to_period('M')
        monthly = df.groupby('month')['fundingRate'].mean().reset_index()
        monthly['symbol'] = symbol
        monthly_data.append(monthly)

    return pd.concat(monthly_data, ignore_index=True)

monthly_df = calc_monthly_funding(funding_data)

# Plot monthly funding for major coins
if len(available_major) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))

    for symbol in available_major[:3]:
        symbol_data = monthly_df[monthly_df['symbol'] == symbol]
        ax.plot(symbol_data['month'].astype(str), symbol_data['fundingRate'] * 100,
                label=symbol, marker='o', markersize=3)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Funding Rate (%)')
    ax.set_title('Monthly Average Funding Rates')
    ax.legend()
    plt.xticks(rotation=45, ha='right')

    # Show every 6th tick to avoid crowding
    tick_positions = range(0, len(symbol_data), 6)
    ax.set_xticks([ax.get_xticks()[i] for i in tick_positions if i < len(ax.get_xticks())])

    plt.tight_layout()
    plt.show()

#%% Summary
print("\n" + "="*60)
print("FUNDING RATE ANALYSIS SUMMARY")
print("="*60)
print(f"\nDataset Overview:")
print(f"  - Total symbols analyzed: {len(funding_data)}")
print(f"  - Total records: {len(combined_df):,}")
print(f"  - Date range: {combined_df['fundingDateTime'].min().date()} to {combined_df['fundingDateTime'].max().date()}")

print(f"\nTop 5 by Annualized Funding Rate:")
for _, row in stats_df.head(5).iterrows():
    print(f"  {row['symbol']}: {row['annualized_rate']:.2f}%")

print(f"\nBottom 5 by Annualized Funding Rate:")
for _, row in stats_df.tail(5).iterrows():
    print(f"  {row['symbol']}: {row['annualized_rate']:.2f}%")
# %%
