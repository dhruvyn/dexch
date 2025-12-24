import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gc


from functools import reduce



def data_check_structured(df):
    results = {}
    
    # Ensure BAR_TIMESTAMP is datetime
    # df['BAR_TIMESTAMP'] = pd.to_datetime(df['BAR_TIMESTAMP'], errors='coerce')
    # df = df.sort_values('BAR_TIMESTAMP').reset_index(drop=True)

    # --- 1. NaN Analysis DataFrame ---
    nan_data = []
    for col in df.columns:
        is_nan = df[col].isna()
        nan_count = is_nan.sum()
        nan_percent = 100 * nan_count / len(df)
        longest_nan = is_nan.groupby((~is_nan).cumsum()).sum().max() if is_nan.any() else 0
        
        nan_data.append({
            "column": col,
            "nan_count": nan_count,
            "nan_percent": round(nan_percent, 2),
            "longest_consecutive_nan": longest_nan
        })
    results['nan_analysis'] = pd.DataFrame(nan_data)

    # --- 2. Timestamp Consistency ---
    df["delta_minutes"] = df["BAR_TIMESTAMP"].diff().dt.total_seconds() / 60
    missing_intervals = df[df["delta_minutes"] != 5].copy()
    
    results['timestamp_summary'] = {
        "total_rows": len(df),
        "duplicate_timestamps": int(df['BAR_TIMESTAMP'].duplicated().sum()),
        "non_5min_count": len(missing_intervals),
        "non_5min_percent": round(100 * len(missing_intervals) / len(df), 2), 
        "non_5min_max":   missing_intervals["delta_minutes"].max()   ,
        "non_5min_min":   missing_intervals["delta_minutes"].min()   ,
        "non_5min_mean": missing_intervals["delta_minutes"].mean() ,
        "non_5min_std":   missing_intervals["delta_minutes"].std()    
    }
    # Store the actual gaps if they exist
    results['missing_intervals_df'] = missing_intervals[["BAR_TIMESTAMP", "delta_minutes"]]

    # --- 3. Numeric Column Statistics ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std']).T
        stats['range'] = stats['max'] - stats['min']
        results['numeric_stats'] = stats.round(4)
    else:
        results['numeric_stats'] = pd.DataFrame()

    return results

def plot_data(df, nan_temporal_plot_only= False) : 
    df['BAR_TIMESTAMP'] = pd.to_datetime(df['BAR_TIMESTAMP'], errors='coerce')
    df = df.sort_values("BAR_TIMESTAMP").reset_index(drop=True)


    # --- 3. NaNs per row over time (smoothed) ---
    nan_counts = df.isna().sum(axis=1)
    rolling_window = 50  # smooth over 50 rows
    nan_counts_smoothed = nan_counts.rolling(rolling_window).mean()
    plt.figure(figsize=(15,4))
    plt.plot(df["BAR_TIMESTAMP"], nan_counts_smoothed)
    plt.title(f"Smoothed NaNs per row over time (rolling window={rolling_window})")
    plt.xlabel("Timestamp")
    plt.ylabel("NaNs (smoothed)")
    plt.show() 
    if nan_temporal_plot_only : 
        return



    # --- 1. NaN heatmap with row grouping ---
    group_size = 200
    grouped_nan = df.isna().groupby(np.arange(len(df)) // group_size).mean()

    plt.figure(figsize=(15,6))
    sns.heatmap(grouped_nan.T, cbar=True, cmap="Reds")
    plt.title(f"NaN Heatmap (smoothed by {group_size} rows)")
    plt.xlabel("Row group")
    plt.ylabel("Columns")
    plt.show()

    # --- 2. Timestamp gaps ---
    df["delta_minutes"] = df["BAR_TIMESTAMP"].diff().dt.total_seconds() / 60

    plt.figure(figsize=(15,4))
    plt.plot(df["BAR_TIMESTAMP"], df["delta_minutes"], marker='o', linestyle='None', markersize=2)
    plt.title("Time Gaps Between Consecutive Rows (minutes)")
    plt.xlabel("Timestamp")
    plt.ylabel("Delta (minutes)")
    plt.ylim(0, df["delta_minutes"].max()+5)
    plt.show()

    # --- 2a. Histogram + aggregated stats for delta_minutes ---
    plt.figure(figsize=(10,4))
    sns.histplot(df["delta_minutes"].dropna(), bins=50, kde=True)
    plt.title("Histogram of delta_minutes (consecutive time gaps)")
    plt.xlabel("Delta minutes")
    plt.ylabel("Frequency")
    plt.show()

    # Aggregated stats
    delta_stats = df["delta_minutes"].describe()
    print("=== delta_minutes Statistics ===")
    print(delta_stats)

    # Optional: count of rows with delta != 5
    non_5min = (df["delta_minutes"] != 5).sum()
    print(f"Number of rows with delta != 5 minutes: {non_5min}")

df_liq = pd.read_csv("../sample_data/btc_5min_liquidations.csv", parse_dates=["BAR_TIMESTAMP"]).sort_values("BAR_TIMESTAMP").reset_index(drop=True)



print(df_liq.columns.tolist())
print(df_liq.head(10))

out = data_check_structured(df_liq)
print(out['timestamp_summary'])
print(out['nan_analysis'])
print(out['numeric_stats'])
plot_data(df_liq)

#set all NaNs to 0s , NaNs only present by mistake in buy/sell avg liq size
df_liq.fillna(0, inplace=True)

assert df_liq.isna().sum().sum() == 0   


df_deriv = pd.read_csv("../sample_data/btc_5min_derivatives.csv", parse_dates=["BAR_TIMESTAMP"]).sort_values("BAR_TIMESTAMP").reset_index(drop=True)



print(df_deriv.columns.tolist())
print(df_deriv.head(10))

out = data_check_structured(df_deriv)
print(out['timestamp_summary'])
print(out['nan_analysis'])
print(out['numeric_stats'])
plot_data(df_deriv)


df_trades = pd.read_csv("../sample_data/btc_5min_trades.csv", parse_dates=["BAR_TIMESTAMP"]).sort_values("BAR_TIMESTAMP").reset_index(drop=True)

print(df_trades.columns.tolist())
print(df_trades.head(10))

out = data_check_structured(df_trades)
print(out['timestamp_summary'])
print(out['nan_analysis'])
print(out['numeric_stats'])
plot_data(df_trades)


# Example: array of DataFrames
dfs = [df_deriv, df_trades, df_liq]  # your list of DataFrames

# Merge all DataFrames on BAR_TIMESTAMP using outer join
df_merged = reduce(lambda left, right: pd.merge(left, right, on="BAR_TIMESTAMP", how="outer"), dfs)

# Sort by timestamp
df_merged = df_merged.sort_values("BAR_TIMESTAMP").reset_index(drop=True)
print(df_merged.head(5) )
print(df_merged.isna().sum().to_string())

out = data_check_structured(df_merged)
print(out['timestamp_summary'])
print(out['nan_analysis'])
print(out['numeric_stats'])

plot_data(df_merged)


# ==============================================================================
# PHASE 0.1: MERGE ARTIFACT CLEANUP
# ==============================================================================
# 1. Drop duplicate/high-NaN columns
cols_to_drop = [
    'EXCHANGE', 'SYMBOL',               
    'EXCHANGE_y', 'SYMBOL_y',           
    'EXCHANGE.1', 'SYMBOL.1',
    'delta_minutes_y', 'delta_minutes_x', 'delta_minutes'     
]
existing_drop = [c for c in cols_to_drop if c in df_merged.columns]
df_merged.drop(columns=existing_drop, inplace=True)

# 2. Rename valid columns
rename_map = {  
    'EXCHANGE_x': 'EXCHANGE',
    'SYMBOL_x': 'SYMBOL'
}
df_merged.rename(columns=rename_map, inplace=True)

# ==============================================================================
# PHASE 0.2: CORE MARKET DATA CLEANUP
# ==============================================================================
core_price_cols = [
    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',       
    'VWAP', 'AVG_PRICE',                                          
    'MARK_PRICE_LAST', 'INDEX_PRICE_LAST', 'LAST_PRICE_LAST',     
    'OPEN_INTEREST_LAST',
    'FUNDING_RATE_LAST'  
]
core_vol_cols = [
    'VOLUME', 'BUY_VOLUME', 'SELL_VOLUME', 
    'NOTIONAL_VOLUME', 'BUY_NOTIONAL', 'SELL_NOTIONAL',
    'TRADES_COUNT', 'TOTAL_TRADES_COUNT', 'BUY_TRADES_COUNT', 'SELL_TRADES_COUNT',
    'TICK_COUNT',      
    'PRICE_STDDEV', 'TRADE_SIZE_STDDEV',
    'AVG_TRADE_SIZE', 'MIN_TRADE_SIZE', 'MAX_TRADE_SIZE',
    'VOLUME_IMBALANCE', 'TRADE_COUNT_IMBALANCE'
]
liq_flow_cols = [
    'BUY_LIQUIDATION_VOLUME', 'SELL_LIQUIDATION_VOLUME', 'TOTAL_LIQUIDATION_VOLUME',
    'BUY_LIQUIDATION_COUNT', 'SELL_LIQUIDATION_COUNT', 'TOTAL_LIQUIDATION_COUNT',
    'BUY_LIQUIDATION_NOTIONAL', 'SELL_LIQUIDATION_NOTIONAL', 'TOTAL_LIQUIDATION_NOTIONAL',
    'AVG_BUY_LIQUIDATION_SIZE', 'AVG_SELL_LIQUIDATION_SIZE', 'AVG_LIQUIDATION_SIZE',
    'MAX_SINGLE_LIQUIDATION_SIZE', 'MAX_SINGLE_LIQUIDATION_NOTIONAL',
    'LIQUIDATION_VOLUME_IMBALANCE', 'LIQUIDATION_COUNT_IMBALANCE'
]
liq_price_cols = [
    'AVG_LIQUIDATION_PRICE', 'VWAP_LIQUIDATION_PRICE', 
    'MIN_LIQUIDATION_PRICE', 'MAX_LIQUIDATION_PRICE'
]
# A. Prices & States -> Forward Fill
existing_price_cols = [c for c in core_price_cols if c in df_merged.columns]
df_merged[existing_price_cols] = df_merged[existing_price_cols].ffill()

# B. Volumes, Counts & Statistics -> Fill with 0
existing_vol_cols = [c for c in core_vol_cols if c in df_merged.columns]
df_merged[existing_vol_cols] = df_merged[existing_vol_cols].fillna(0)

# ==============================================================================
# PHASE 0.3: LIQUIDATION DATA CLEANUP
# ==============================================================================

# Group A: Liquidation Flows (Events) -> Fill with 0
existing_liq_flow = [c for c in liq_flow_cols if c in df_merged.columns]
df_merged[existing_liq_flow] = df_merged[existing_liq_flow].fillna(0)

# Group B: Liquidation Levels (Prices) -> Forward Fill
existing_liq_price = [c for c in liq_price_cols if c in df_merged.columns]
df_merged[existing_liq_price] = df_merged[existing_liq_price].ffill()

# ==============================================================================
# PHASE 0.4: FINAL SAFETY CHECK
# ==============================================================================
# Drop rows where ffill failed at the start
df_merged.dropna(subset=existing_price_cols, inplace=True)

# Verification
print("Final NaN Check (Should only show timestamps):")
print(df_merged.isna().sum()[df_merged.isna().sum() > 0])
print(f"Final Shape: {df_merged.shape}")



# ==========================================
# PHASE 0.5: TIMESTAMP ENGINEERING & FINAL POLISH
# ==========================================

# 0. Convert to DateTime Objects
for col in df_merged.columns :
    if "TIMESTAMP" in col:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')


# 1. Harvest "Velocity" & "Span" Features
# ------------------------------------------------
# We extract signals from ALL timestamp columns before dropping them.

# A. Liquidation Velocity
# LIQ_DURATION: Flash crash (0s) vs Slow bleed (high values).
df_merged['LIQ_DURATION'] = (
    df_merged['LAST_LIQUIDATION_TIMESTAMP'] - df_merged['FIRST_LIQUIDATION_TIMESTAMP']
).dt.total_seconds().fillna(0)

# LIQ_LATENCY: Did it happen at the start or end of the bar? (Momentum signal)
df_merged['LIQ_LATENCY'] = (
    df_merged['LAST_LIQUIDATION_TIMESTAMP'] - df_merged['BAR_TIMESTAMP']
).dt.total_seconds().fillna(0)

# B. Trade Liquidity Span
# TRADE_SPAN: Did trading occur continuously (300s) or in a single burst (<1s)?
df_merged['TRADE_SPAN'] = (
    df_merged['LAST_TRADE_TIMESTAMP'] - df_merged['FIRST_TRADE_TIMESTAMP']
).dt.total_seconds().fillna(0)

# C. Tick Microstructure Span
# TICK_SPAN: Granular measure of order book activity duration.
df_merged['TICK_SPAN'] = (
    df_merged['LAST_TICK_TIMESTAMP'] - df_merged['FIRST_TICK_TIMESTAMP']
).dt.total_seconds().fillna(0)

# D. Funding Seasonality
# SECONDS_SINCE_FUNDING: Proximity to the 8-hour funding payout/squeeze.
df_merged['SECONDS_SINCE_FUNDING'] = (
    df_merged['BAR_TIMESTAMP'] - df_merged['FUNDING_TIMESTAMP_LAST']
).dt.total_seconds().fillna(0)

# 2. Drop Raw Metadata Columns
# ------------------------------------------------
ts_cols_to_drop = [
    'FIRST_LIQUIDATION_TIMESTAMP', 'LAST_LIQUIDATION_TIMESTAMP',
    'FIRST_TRADE_TIMESTAMP', 'LAST_TRADE_TIMESTAMP',
    'FIRST_TICK_TIMESTAMP', 'LAST_TICK_TIMESTAMP',
    'FUNDING_TIMESTAMP_LAST' 
]
existing_ts_drop = [c for c in ts_cols_to_drop if c in df_merged.columns]
df_merged.drop(columns=existing_ts_drop, inplace=True)


# 3. Final Edge Case Cleanup (The "1 NaN" Row)
# ------------------------------------------------
liq_price_cols = [
    'AVG_LIQUIDATION_PRICE', 'VWAP_LIQUIDATION_PRICE', 
    'MIN_LIQUIDATION_PRICE', 'MAX_LIQUIDATION_PRICE'
]
# Only check subset if columns exist
existing_liq_checks = [c for c in liq_price_cols if c in df_merged.columns]
if existing_liq_checks:
    df_merged.dropna(subset=existing_liq_checks, inplace=True)




# 4. Final Verification
# ------------------------------------------------
print("--- FINAL DATASET STATUS ---")
print(f"Total Rows: {len(df_merged)}")
print(f"Columns: {len(df_merged.columns)}")
print("Remaining NaNs (Must be 0):")
print(df_merged.isna().sum().sum())

# Preview
print("\nNew Features Sample:")
print(df_merged[['LIQ_DURATION', 'TRADE_SPAN']].head())



out = data_check_structured(df_merged)
print(out['timestamp_summary'])
print(out['nan_analysis'])
print(out['numeric_stats'])

plot_data(df_merged)