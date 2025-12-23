import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import os   
import subprocess
import sys

# result based editing of buckets
bucket_a_cols = [
    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',
    'VWAP', 'AVG_PRICE',
    'MARK_PRICE_LAST', 'INDEX_PRICE_LAST', 'LAST_PRICE_LAST'
]
# REMOVED: OPEN_INTEREST_LAST (Moved to B)
bucket_b_cols = [
    # Moved Here:
    'OPEN_INTEREST_LAST', 

    # Existing:
    'VOLUME', 'BUY_VOLUME', 'SELL_VOLUME',
    'NOTIONAL_VOLUME', 'BUY_NOTIONAL', 'SELL_NOTIONAL',
    'TICK_COUNT', 'TOTAL_TRADES_COUNT', 'BUY_TRADES_COUNT', 'SELL_TRADES_COUNT',

    # Liquidation Alpha (Massive Kurtosis confirmed)
    'BUY_LIQUIDATION_VOLUME', 'SELL_LIQUIDATION_VOLUME', 'TOTAL_LIQUIDATION_VOLUME',
    'BUY_LIQUIDATION_COUNT', 'SELL_LIQUIDATION_COUNT', 'TOTAL_LIQUIDATION_COUNT',
    'BUY_LIQUIDATION_NOTIONAL', 'SELL_LIQUIDATION_NOTIONAL', 'TOTAL_LIQUIDATION_NOTIONAL',

    # Sizes
    'AVG_TRADE_SIZE', 'MIN_TRADE_SIZE', 'MAX_TRADE_SIZE',
    'AVG_BUY_LIQUIDATION_SIZE', 'AVG_SELL_LIQUIDATION_SIZE', 'AVG_LIQUIDATION_SIZE',
    'MAX_SINGLE_LIQUIDATION_SIZE', 'MAX_SINGLE_LIQUIDATION_NOTIONAL',
    'PRICE_STDDEV', 'TRADE_SIZE_STDDEV',

    # Velocity
    'LIQ_DURATION', 'LIQ_LATENCY', 
    'TRADE_SPAN', 'TICK_SPAN'
]
# REMOVED: SECONDS_SINCE_FUNDING (Moved to C)
bucket_c_cols = [
    # Moved Here:
    'SECONDS_SINCE_FUNDING', # It's a periodic cycle (Oscillator)

    # Existing:
    'FUNDING_RATE_LAST',
    'VOLUME_IMBALANCE', 'TRADE_COUNT_IMBALANCE',
    'LIQUIDATION_VOLUME_IMBALANCE', 'LIQUIDATION_COUNT_IMBALANCE',

    # Liquidation Levels (MUST be transformed to Spreads)
    'MIN_LIQUIDATION_PRICE', 'MAX_LIQUIDATION_PRICE', 
    'AVG_LIQUIDATION_PRICE', 'VWAP_LIQUIDATION_PRICE'
]

# ==============================================================================
# STEP 1: TRANSFORM LIQUIDATION PRICES -> STATIONARY SPREADS
# ==============================================================================
# 1. Define the targets (Raw Level -> Distance)
# We map the raw column name to a new feature name
liq_transform_map = {
    'MIN_LIQUIDATION_PRICE': 'DIST_MIN_LIQ',
    'MAX_LIQUIDATION_PRICE': 'DIST_MAX_LIQ',
    'AVG_LIQUIDATION_PRICE': 'DIST_AVG_LIQ',
    'VWAP_LIQUIDATION_PRICE': 'DIST_VWAP_LIQ'
}

def transform_liq_prices_update_inplace(df, verbose= True) : 
    print("--- TRANSFORMING LIQUIDATION PRICES ---")

    for raw_col, new_col in liq_transform_map.items():
        if raw_col in df.columns:
            # A. Calculate Log-Spread: Log(Liq) - Log(Close)
            # This creates a stationary "percentage distance" metric
            df[new_col] = np.log(df[raw_col]) - np.log(df['CLOSE_PRICE'])
            
            # B. Handle NaNs (just in case any slipped through, though we cleaned them)
            df[new_col] = df[new_col].fillna(0)
            
            # C. Drop the original Raw Price column (it's dangerous for ML)
            df.drop(columns=[raw_col], inplace=True)
            if verbose: 
                print(f"Created {new_col} | Dropped {raw_col}")
    
    return 

def get_weights_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d=0.4, thres=1e-5):
    # 1. Compute weights (w is returned as [w_K, ..., w_0])
    w = get_weights_ffd(d, thres).flatten()
    width = len(w)
    
    # 2. Filter valid data (remove existing NaNs to process)
    # We maintain indices to map back to the original series
    series_clean = series.dropna()
    if len(series_clean) < width:
        return pd.Series(index=series.index, dtype=float)
        
    # 3. Create a windowed view of the data (N, width)
    # This creates a view, not a copy, so it is memory efficient
    windows = sliding_window_view(series_clean.values, window_shape=width)
    
    # 4. Apply dot product vectorized
    # windows shape: (n_obs - width + 1, width)
    # w shape: (width,)
    result_values = np.dot(windows, w)
    
    # 5. Realign with index
    # The result corresponds to the timestamps starting from series_clean.index[width-1]
    result_index = series_clean.index[width-1:]
    
    return pd.Series(result_values, index=result_index).reindex(series.index)

# ==============================================================================
# BUCKET A PROCESS (Log -> FracDiff -> Rolling Z-Score)
# ==============================================================================
def bucket_a_process(df, cols, verbose= True,  meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], 
                     window=2000, d=0.4):
    if verbose :
        print("\n--- PROCESSING BUCKET A (Trend) ---")
    
    # 1. Create Isolated DataFrame
    df_a = df[meta_cols + cols].copy()
    
    # 2. Apply Transformations
    for col in cols:
        if col not in df_a.columns:
            print("Skipping", col)
            continue
        if verbose :
            print(f"Processing {col}...")
        
        # A. Log Transform
        df_a[col] = np.log(df_a[col] + 1e-9)
        
        # B. FracDiff
        df_a[col] = frac_diff_ffd(df_a[col], d=d)
        
        # C. Rolling Z-Score
        rolling_mean = df_a[col].rolling(window=window).mean()
        rolling_std = df_a[col].rolling(window=window).std()
        df_a[col] = (df_a[col] - rolling_mean) / (rolling_std + 1e-9)

    # 3. Rename Columns (Add suffix '_A')
    # We create a dictionary mapping old_name -> old_name_A
    rename_map = {col: f"{col}_A" for col in cols}
    df_a.rename(columns=rename_map, inplace=True)
    
    if verbose : 
        print(f"Bucket A Complete. Shape: {df_a.shape}")
    return df_a

# # ==============================================================================
# # BUCKET B PROCESS (Log1p -> Rolling Robust Scaler -> Rename)
# # ==============================================================================
def bucket_b_process(df, cols, verbose= True,  meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], 
                      window=2000):
    if verbose : 
        print("\n--- PROCESSING BUCKET B (Unified Robust Scaler) ---")
    
    valid_cols = [c for c in cols if c in df.columns]
    df_b = df[meta_cols + valid_cols].copy()
    
    for col in valid_cols:
        if col not in df_b.columns:
            print("Skipping", col)
            continue
        if verbose : 
            print(f"Processing {col}...")
        
        # 1. Log Transform (Always good for volume/notional)
        # Compresses the massive 100x spikes so they are manageable
        df_b[col] = np.log1p(df_b[col])
        
        # 2. Rolling Stats
        rolling_median = df_b[col].rolling(window=window).median()
        rolling_q75 = df_b[col].rolling(window=window).quantile(0.75)
        rolling_q25 = df_b[col].rolling(window=window).quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25
        
        # 3. THE FIX: "Safety Floor" on IQR
        # If IQR is 0 (because data is sparse/constant), we enforce a minimum scale.
        # We use a tiny static epsilon (1e-6) OR 1% of the median, whichever is safer.
        
        # This prevents the "Explosion" we saw earlier.
        iqr_floor = np.maximum(rolling_median * 0.01, 1e-6)
        safe_iqr = np.maximum(rolling_iqr, iqr_floor)
        
        # 4. Apply Robust Scaling
        # Result: Centered around 0. Most data between -1 and 1.
        # Outliers will be large (e.g., 5, 10) but won't ruin the rest.
        df_b[col] = (df_b[col] - rolling_median) / safe_iqr
        
        # Fill NaNs (created by the rolling window) with 0
        df_b[col] = df_b[col].fillna(0)
    if verbose :
        print(f"Bucket B Complete. Shape: {df_b.shape}")
    # Rename
    rename_map = {col: f"{col}_B" for col in valid_cols}
    df_b.rename(columns=rename_map, inplace=True)
    
    return df_b

# ==============================================================================
# BUCKET C PROCESS (Rolling Winsorize -> Rolling MinMax -> Rename)
# ==============================================================================
def bucket_c_process(df, cols, verbose= True, meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], 
                     window=2000):
    if verbose :
        print("\n--- PROCESSING BUCKET C (Oscillators) ---")
    
    valid_cols = [c for c in cols if c in df.columns]
    df_c = df[meta_cols + valid_cols].copy()
    
    for col in valid_cols:
        if col not in df_c.columns:
            print("Skipping", col)
            continue
        if verbose : 
            print(f"Processing {col}...")
        
        # 1. Rolling Winsorization (Clipping)
        # We calculate the rolling 1st and 99th percentiles
        roll_p01 = df_c[col].rolling(window=window).quantile(0.01)
        roll_p99 = df_c[col].rolling(window=window).quantile(0.99)
        
        # Clip the data to these dynamic bounds
        # We use .clip() with the rolling series (aligned by index)
        # Note: 'clip' with series arguments requires matching indices
        df_c[col] = df_c[col].clip(lower=roll_p01, upper=roll_p99, axis=0)
        
        # 2. Rolling MinMax Scaler
        # Now we scale based on the clipped range
        roll_min = df_c[col].rolling(window=window).min()
        roll_max = df_c[col].rolling(window=window).max()
        roll_range = roll_max - roll_min
        
        # Apply Scaling (Result is 0.0 to 1.0)
        df_c[col] = (df_c[col] - roll_min) / (roll_range + 1e-9)

    # 3. Rename Columns (Add suffix '_C')
    rename_map = {col: f"{col}_C" for col in valid_cols}
    df_c.rename(columns=rename_map, inplace=True)
    if verbose :
        print(f"Bucket C Complete. Shape: {df_c.shape}")
    return df_c

def merge(df_bucket_a, df_bucket_b, df_bucket_c, verbose= True):
    
    # ==============================================================================
    # PHASE 3: FINAL MERGE & CLEANUP
    # ==============================================================================
    if verbose:
        print("--- STARTING FINAL MERGE ---")

    # 1. Merge Bucket A + Bucket B
    # Inner join ensures we only keep rows that exist in both (safety check)
    df_final = pd.merge(
        df_bucket_a, 
        df_bucket_b, 
        on=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], 
        how='inner'
    )

    # 2. Merge Result + Bucket C
    df_final = pd.merge(
        df_final, 
        df_bucket_c, 
        on=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], 
        how='inner'
    )

    # 3. Final Cleanup: Drop "Warmup" NaNs
    # Our rolling windows (2000 bars) created NaNs at the start of the dataset.
    # We must drop these rows so the model doesn't see garbage.
    rows_before = len(df_final)
    df_final.dropna(inplace=True)
    rows_dropped = rows_before - len(df_final)

    
    # ==============================================================================
    # FINAL VERIFICATION
    # ==============================================================================
    if verbose : 
        print(f"\n--- MERGE COMPLETE ---")
        print(f"Original Rows:  {rows_before}")
        print(f"Dropped (Warmup): {rows_dropped} (Matches your window size?)")
        print(f"Final Shape:    {df_final.shape}")
        print(f"Total Features: {len(df_final.columns) - 2}") # Excluding Symbol/Exchange

        # Check for any lingering NaNs
        nan_count = df_final.isna().sum().sum()
        if nan_count == 0:
            print("Status: SUCCESS (0 NaNs found. Dataset is fully dense.)")
        else:
            print(f"Status: WARNING ({nan_count} NaNs remain!)")

        # Preview
        print("\nFeature Matrix Preview:")
        print(df_final.iloc[:5, :5]) # Show first 5 rows/cols


    # ==========================================
    # PHASE 3.5: POST-TRANSFORM CLEANUP (FIXED)
    # ==========================================

    # 1. Identify Numeric Columns First
    # We only calculate std() on actual numbers to avoid string errors
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns

    # 2. Identify Constant Columns (Zero Variance) within numeric cols
    # We assume if std() is 0 (or NaN/Null for safety), it provides no signal.
    # Add this cleanup step after your bucket processing
    cols_to_drop = ['MIN_TRADE_SIZE_B', 'TRADE_SPAN_B']
    if verbose : 
        print(f"Dropping noise features: {cols_to_drop}")
    df_final.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # 4. Final Shape Check
    if verbose :
        print(f"Final Clean Matrix Shape: {df_final.shape}")


    return df_final


def run_X_pipeline(df, verbose= True):
    # --- EXECUTE ---
    df_bucket_c = bucket_c_process(df, bucket_c_cols, verbose)
    df_bucket_b = bucket_b_process(df, bucket_b_cols, verbose)
    df_bucket_a = bucket_a_process(df, bucket_a_cols, verbose)


    # Column Check

    missing_a = [a for a in bucket_a_cols if (a+"_A") not in df_bucket_a.columns.tolist()]
    missing_b = [b for b in bucket_b_cols if (b+"_B") not in df_bucket_b.columns.tolist()]
    missing_c = [c for c in bucket_c_cols if (c+"_C") not in df_bucket_c.columns.tolist()]
    missing_cols = [v for v in bucket_a_cols + bucket_b_cols + bucket_c_cols if v not in df.columns.tolist()]
    assert len(missing_a) == 0
    assert len(missing_b) == 0  
    assert len(missing_c) == 0
    assert len(missing_cols) == 0


    df_final = merge(df_bucket_a, df_bucket_b, df_bucket_c, verbose)
    return df_final

if __name__ == "__main__":
    # arg parse to get folder name to dump each split in a new subdirectory inside the folde
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, help="Folder name to dump each split in a new subdirectory inside the folder")
    parser.add_argument("--num_splits", type=int, help="Number of splits to create")
    args = parser.parse_args()
    folder_name = args.folder_name
    num_splits = args.num_splits    

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # Load Data
    df = pd.read_csv("./sample_data/btc_5min_nan_processed_data.csv", parse_dates=["BAR_TIMESTAMP"])

    df.sort_values(["BAR_TIMESTAMP"], inplace=True)

    target = pd.read_csv("./sample_data/btc_5min_targets.csv")


    print("Bucket A List: ")
    print(bucket_a_cols)
    print("Bucket B List: ")
    print(bucket_b_cols)
    #Update the Python List 'bucket_c_cols'
    # We remove the old names and add the new 'DIST_' names so your list stays valid.
    bucket_c_cols = [col for col in bucket_c_cols if col not in liq_transform_map.keys()]
    bucket_c_cols.extend(liq_transform_map.values())
    print("\nUpdated Bucket C List:")
    print(bucket_c_cols)


    def generate_k_splits(n, k):
        indices = np.arange(n)
        return np.array_split(indices, k)

    # usage
    n = len(df)
    splits = generate_k_splits(n, k=num_splits)


    # index dataframe
    for split in splits:
        df_split = df.iloc[split]

        transform_liq_prices_update_inplace(df_split, vebose= True)

        start= split[0], end = split[-1]
        X_path= f"{folder_name}/X_{split[0]}_{split[-1]}.csv"

        X = run_X_pipeline(df_split, verbose = True)

        #dump to csv
        X.to_csv(X_path, index=False)



        # run another python script to run the analysis on the split
        # give the script, the path to X and the split for the y as args to argpars
        # run the subprocess
        RUN_FILE = "run_analysis_refactor.py"
        subprocess.run(
            [
                sys.executable,      # ensures same python env
                RUN_FILE,
                "--X_path", X_path,
                "--start_idx", str(start),
                "--end_idx", str(end),
            ],
            check=True
        )






        

