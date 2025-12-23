import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import os
import subprocess
import sys

# ==========================================
# LOGGING UTILITY
# ==========================================
class Logger(object):
    """Duplicates stdout to a file and the console."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ==========================================
# CONFIGURATION: FEATURE BUCKETS
# ==========================================
bucket_a_cols = [
    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',
    'VWAP', 'AVG_PRICE',
    'MARK_PRICE_LAST', 'INDEX_PRICE_LAST', 'LAST_PRICE_LAST'
]

bucket_b_cols = [
    'OPEN_INTEREST_LAST', 
    'VOLUME', 'BUY_VOLUME', 'SELL_VOLUME',
    'NOTIONAL_VOLUME', 'BUY_NOTIONAL', 'SELL_NOTIONAL',
    'TICK_COUNT', 'TOTAL_TRADES_COUNT', 'BUY_TRADES_COUNT', 'SELL_TRADES_COUNT',
    'BUY_LIQUIDATION_VOLUME', 'SELL_LIQUIDATION_VOLUME', 'TOTAL_LIQUIDATION_VOLUME',
    'BUY_LIQUIDATION_COUNT', 'SELL_LIQUIDATION_COUNT', 'TOTAL_LIQUIDATION_COUNT',
    'BUY_LIQUIDATION_NOTIONAL', 'SELL_LIQUIDATION_NOTIONAL', 'TOTAL_LIQUIDATION_NOTIONAL',
    'AVG_TRADE_SIZE', 'MIN_TRADE_SIZE', 'MAX_TRADE_SIZE',
    'AVG_BUY_LIQUIDATION_SIZE', 'AVG_SELL_LIQUIDATION_SIZE', 'AVG_LIQUIDATION_SIZE',
    'MAX_SINGLE_LIQUIDATION_SIZE', 'MAX_SINGLE_LIQUIDATION_NOTIONAL',
    'PRICE_STDDEV', 'TRADE_SIZE_STDDEV',
    'LIQ_DURATION', 'LIQ_LATENCY', 
    'TRADE_SPAN', 'TICK_SPAN'
]

bucket_c_cols = [
    'SECONDS_SINCE_FUNDING',
    'FUNDING_RATE_LAST',
    'VOLUME_IMBALANCE', 'TRADE_COUNT_IMBALANCE',
    'LIQUIDATION_VOLUME_IMBALANCE', 'LIQUIDATION_COUNT_IMBALANCE',
    'MIN_LIQUIDATION_PRICE', 'MAX_LIQUIDATION_PRICE', 
    'AVG_LIQUIDATION_PRICE', 'VWAP_LIQUIDATION_PRICE'
]

# Transform Map for Liquidation Levels
liq_transform_map = {
    'MIN_LIQUIDATION_PRICE': 'DIST_MIN_LIQ',
    'MAX_LIQUIDATION_PRICE': 'DIST_MAX_LIQ',
    'AVG_LIQUIDATION_PRICE': 'DIST_AVG_LIQ',
    'VWAP_LIQUIDATION_PRICE': 'DIST_VWAP_LIQ'
}

# ==========================================
# HELPER FUNCTIONS: TRANSFORMATIONS
# ==========================================

def transform_liq_prices_update_inplace(df, verbose=True): 
    if verbose: print("--- TRANSFORMING LIQUIDATION PRICES ---")

    for raw_col, new_col in liq_transform_map.items():
        if raw_col in df.columns:
            # Log-Spread: Log(Liq) - Log(Close)
            df[new_col] = np.log(df[raw_col]) - np.log(df['CLOSE_PRICE'])
            df[new_col] = df[new_col].fillna(0)
            df.drop(columns=[raw_col], inplace=True)
            
            if verbose: print(f"Created {new_col} | Dropped {raw_col}")

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
    w = get_weights_ffd(d, thres).flatten()
    width = len(w)
    
    series_clean = series.dropna()
    if len(series_clean) < width:
        return pd.Series(index=series.index, dtype=float)
        
    windows = sliding_window_view(series_clean.values, window_shape=width)
    result_values = np.dot(windows, w)
    
    result_index = series_clean.index[width-1:]
    return pd.Series(result_values, index=result_index).reindex(series.index)

# ==========================================
# PIPELINE: BUCKET PROCESSORS
# ==========================================

def bucket_a_process(df, cols, verbose=True, meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], window=2000, d=0.4):
    if verbose: print("\n--- PROCESSING BUCKET A (Trend) ---")
    
    df_a = df[meta_cols + [c for c in cols if c in df.columns]].copy()
    
    for col in cols:
        if col not in df_a.columns: continue
        if verbose: print(f"Processing {col}...")
        
        # Log -> FracDiff -> Rolling Z-Score
        df_a[col] = np.log(df_a[col] + 1e-9)
        df_a[col] = frac_diff_ffd(df_a[col], d=d)
        
        rolling_mean = df_a[col].rolling(window=window).mean()
        rolling_std = df_a[col].rolling(window=window).std()
        df_a[col] = (df_a[col] - rolling_mean) / (rolling_std + 1e-9)

    rename_map = {col: f"{col}_A" for col in cols if col in df_a.columns}
    df_a.rename(columns=rename_map, inplace=True)
    
    if verbose: print(f"Bucket A Complete. Shape: {df_a.shape}")
    return df_a

def bucket_b_process(df, cols, verbose=True, meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], window=2000):
    if verbose: print("\n--- PROCESSING BUCKET B (Robust Scaler) ---")
    
    valid_cols = [c for c in cols if c in df.columns]
    df_b = df[meta_cols + valid_cols].copy()
    
    for col in valid_cols:
        if verbose: print(f"Processing {col}...")
        
        # Log1p -> Rolling Robust Scaler
        df_b[col] = np.log1p(df_b[col])
        
        rolling_median = df_b[col].rolling(window=window).median()
        rolling_q75 = df_b[col].rolling(window=window).quantile(0.75)
        rolling_q25 = df_b[col].rolling(window=window).quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25
        
        # Safety Floor
        iqr_floor = np.maximum(rolling_median * 0.01, 1e-6)
        safe_iqr = np.maximum(rolling_iqr, iqr_floor)
        
        df_b[col] = (df_b[col] - rolling_median) / safe_iqr
        df_b[col] = df_b[col].fillna(0)

    rename_map = {col: f"{col}_B" for col in valid_cols}
    df_b.rename(columns=rename_map, inplace=True)
    
    if verbose: print(f"Bucket B Complete. Shape: {df_b.shape}")
    return df_b

def bucket_c_process(df, cols, verbose=True, meta_cols=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], window=2000):
    if verbose: print("\n--- PROCESSING BUCKET C (Oscillators) ---")
    
    valid_cols = [c for c in cols if c in df.columns]
    df_c = df[meta_cols + valid_cols].copy()
    
    for col in valid_cols:
        if verbose: print(f"Processing {col}...")
        
        # Rolling Winsorize -> Rolling MinMax
        roll_p01 = df_c[col].rolling(window=window).quantile(0.01)
        roll_p99 = df_c[col].rolling(window=window).quantile(0.99)
        
        df_c[col] = df_c[col].clip(lower=roll_p01, upper=roll_p99, axis=0)
        
        roll_min = df_c[col].rolling(window=window).min()
        roll_max = df_c[col].rolling(window=window).max()
        roll_range = roll_max - roll_min
        
        df_c[col] = (df_c[col] - roll_min) / (roll_range + 1e-9)

    rename_map = {col: f"{col}_C" for col in valid_cols}
    df_c.rename(columns=rename_map, inplace=True)
    
    if verbose: print(f"Bucket C Complete. Shape: {df_c.shape}")
    return df_c

def merge_and_clean(df_bucket_a, df_bucket_b, df_bucket_c, verbose=True):
    if verbose: print("--- STARTING FINAL MERGE ---")

    # 1. Merge all buckets
    df_final = pd.merge(df_bucket_a, df_bucket_b, on=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], how='inner')
    df_final = pd.merge(df_final, df_bucket_c, on=['SYMBOL', 'EXCHANGE', 'BAR_TIMESTAMP'], how='inner')

    # 2. Drop Warmup NaNs (Result of rolling windows)
    rows_before = len(df_final)
    df_final.dropna(inplace=True)
    
    # 3. Post-Transform Cleanup (Drop Noise)
    cols_to_drop = ['MIN_TRADE_SIZE_B', 'TRADE_SPAN_B']
    df_final.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    if verbose: 
        print(f"\n--- MERGE COMPLETE ---")
        print(f"Original Rows:  {rows_before}")
        print(f"Final Shape:    {df_final.shape}")
        
    return df_final

def run_X_pipeline(df, verbose=True):
    # Execute Buckets
    df_bucket_a = bucket_a_process(df, bucket_a_cols, verbose)
    df_bucket_b = bucket_b_process(df, bucket_b_cols, verbose)
    df_bucket_c = bucket_c_process(df, bucket_c_cols, verbose)

    # Sanity Checks
    missing_a = [a for a in bucket_a_cols if (a+"_A") not in df_bucket_a.columns]
    missing_b = [b for b in bucket_b_cols if (b+"_B") not in df_bucket_b.columns]
    missing_c = [c for c in bucket_c_cols if (c+"_C") not in df_bucket_c.columns]
    
    assert len(missing_a) == 0, f"Missing A: {missing_a}"
    assert len(missing_b) == 0, f"Missing B: {missing_b}"
    assert len(missing_c) == 0, f"Missing C: {missing_c}"

    df_final = merge_and_clean(df_bucket_a, df_bucket_b, df_bucket_c, verbose)
    return df_final

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True, help="Output directory root")
    parser.add_argument("--num_splits", type=int, required=True, help="Number of splits")
    args = parser.parse_args()
    
    folder_name = args.folder_name
    num_splits = args.num_splits    

    # 1. Setup Main Directory and Logging
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    log_file_path = os.path.join(folder_name, "log.txt")
    sys.stdout = Logger(log_file_path)

    try:
        # 2. Load Data
        print("Loading Data...")
        FILE_PATH = "./sample_data/btc_5min_nan_processed_data.csv"
        df = pd.read_csv(FILE_PATH, parse_dates=["BAR_TIMESTAMP"])
        df.sort_values(["BAR_TIMESTAMP"], inplace=True)
        
        # 3. Update Bucket C List (Dynamic handling for liq prices)
        bucket_c_cols = [col for col in bucket_c_cols if col not in liq_transform_map.keys()]
        bucket_c_cols.extend(liq_transform_map.values())
        print(f"Updated Bucket C List: {bucket_c_cols}")

        # 4. Generate Splits
        indices = np.arange(len(df))
        splits = np.array_split(indices, num_splits)

        print(f"Starting execution on {num_splits} splits...")

        for i, split in enumerate(splits):
            split_id = i + 1
            print(f"\n{'='*40}")
            print(f"PROCESSING SPLIT {split_id}/{num_splits}")
            print(f"{'='*40}")

            # Create copy to avoid SettingWithCopy warnings
            df_split = df.iloc[split].copy()
            
            # A. Inplace Transformation (Liquidation columns)
            transform_liq_prices_update_inplace(df_split, verbose=True)

            # B. Run Pipeline
            X = run_X_pipeline(df_split, verbose=True)
            
            # C. Define Directory Structure: folder_name/split_N/data_split/
            split_dir = os.path.join(folder_name, f"split_{split_id}")
            data_split_dir = os.path.join(split_dir, "data_split")
            
            if not os.path.exists(data_split_dir):
                os.makedirs(data_split_dir)

            # D. Save Processed Split
            global_start = split[0]
            global_end = split[-1]
            X_path = os.path.join(data_split_dir, f"X_{global_start}_{global_end}.csv")
            
            print(f"Saving split data to: {X_path}")
            X.to_csv(X_path, index=False)

            # E. Run Analysis Subprocess
            # NOTE: subprocess doesn't need start/end args anymore as it reads the full file
            RUN_FILE = "./scripts/run_analysis_refactor.py"
            print(f"Launching subprocess: {RUN_FILE}")
            
        # ... inside your loop ...

            print(f"Launching subprocess: {RUN_FILE}")

            # Use Popen to stream output line-by-line
            with subprocess.Popen(
                [
                    sys.executable,
                    RUN_FILE,
                    "--X_path", X_path,
                    "--split_dir", split_dir
                    # Remove start/end indices as discussed
                ],
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.STDOUT, # Redirect stderr to stdout so errors define logged too
                text=True,               # Decode to string immediately
                bufsize=1                # Line buffering
            ) as proc:
                # Read output line by line as it is generated
                for line in proc.stdout:
                    print(line, end='')  # This triggers your Logger.write() (Terminal + File)

            if proc.returncode != 0:
                print(f"Subprocess failed with return code {proc.returncode}")
                raise subprocess.CalledProcessError(proc.returncode, RUN_FILE)
                
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise
    finally:
        # Stop logging
        sys.stdout.close()
        sys.stdout = sys.__stdout__