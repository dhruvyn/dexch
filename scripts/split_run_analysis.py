import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import os
import subprocess
import sys
import gc  # <--- ADDED: To force memory cleanup
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kurtosis


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
TIME_STAMP_NAME ="BAR_TIMESTAMP"

# ==========================================
# HELPER FUNCTIONS: TRANSFORMATIONS
# ==========================================

def test_bucket_hypothesis(df, bucket_cols, bucket_name):
    print(f"\n--- TESTING {bucket_name} ---")
    results = []
    
    for col in bucket_cols:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        if len(series) < 20: # Skip if empty
            continue
            
        # 1. Stationarity Test (ADF)
        # Null Hypothesis: Series is Non-Stationary.
        # p < 0.05 = Rejects Null (It IS Stationary).
        # p > 0.05 = Fails to Reject (It is Trending/Drifting).
        try:
            adf_result = adfuller(series.values, maxlag=1)
            p_value = adf_result[1]
        except:
            p_value = 1.0 # Default fail

        # 2. Heavy Tail Test (Kurtosis)
        # Normal Distribution = 3.0 (Fisher definition uses 0 as normal, so >0 is heavy)
        # We use Pearson (Normal = 3). Values > 3 imply "Spiky" outliers.
        kurt_val = kurtosis(series, fisher=False) 

        # 3. Sparsity (Zero %)
        zero_pct = (series == 0).mean()

        # --- EVALUATION LOGIC ---
        status = "PASS"
        reason = ""

        if bucket_name == "BUCKET_A":
            # Hypothesis: Should be Non-Stationary (High Memory)
            if p_value < 0.05: 
                status = "WARNING"
                reason = "Already Stationary (Maybe skip FracDiff?)"
        
        elif bucket_name == "BUCKET_B":
            # Hypothesis: Should be Spiky/Heavy Tailed
            if kurt_val < 3:
                status = "NOTE"
                reason = "Low Kurtosis (Not spiky, maybe standard scale?)"
        
        elif bucket_name == "BUCKET_C":
            # Hypothesis: Should be Stationary (Oscillator)
            # EXCEPTION: Liquidation Prices (MIN/MAX) are technically Prices (Drifting)
            # until we transform them to Spreads. We check this specifically.
            if "LIQUIDATION_PRICE" in col:
                # We expect raw prices to fail stationarity, proving they need transformation
                if p_value > 0.05:
                    status = "CONFIRMED"
                    reason = "Raw Price Drifts (Needs 'Spread' transform)"
                else:
                    status = "WEIRD"
                    reason = "Price is stationary??"
            elif p_value > 0.05:
                status = "WARNING"
                reason = "Non-Stationary (Ratio is drifting!)"

        # Append
        results.append({
            "Feature": col,
            "ADF_p_val": round(p_value, 4),
            "Kurtosis": round(kurt_val, 1),
            "Zeros%": round(zero_pct, 2),
            "Status": status,
            "Comment": reason
        })
    
    # Print Report
    report_df = pd.DataFrame(results)
    print(report_df.to_string())

def transform_liq_prices_update_inplace(df, verbose=True): 
    if verbose: print("--- TRANSFORMING LIQUIDATION PRICES ---")
    # log(liq) - log(close) -> fillna(0)

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
        
        # Log1p -> Rolling Robust Scaler, iqr
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
    missing_cols = [v for v in bucket_a_cols + bucket_b_cols + bucket_c_cols if v not in df.columns.tolist()]

    assert len(missing_a) == 0, f"Missing A: {missing_a}"
    assert len(missing_b) == 0, f"Missing B: {missing_b}"
    assert len(missing_c) == 0, f"Missing C: {missing_c}"
    assert len(missing_cols) == 0, f"Missing Columns: {missing_cols}"

    df_final = merge_and_clean(df_bucket_a, df_bucket_b, df_bucket_c, verbose)

    print("------- STARTING HYPOTHESIS TESTS -------")
    # 1. Test Bucket A (Expect High p-values)
    test_bucket_hypothesis(df_final,[a for a in  df_bucket_a.columns.to_list() if ("BAR_TIMESTAMP" not in a) and ("EXCHANGE" not in a) and ("SYMBOL" not in a)], "BUCKET_A")

    # 2. Test Bucket B (Expect High Kurtosis)
    test_bucket_hypothesis(df_final, [a for a in  df_bucket_b.columns.to_list() if ("BAR_TIMESTAMP" not in a) and ("EXCHANGE" not in a) and ("SYMBOL" not in a)], "BUCKET_B")

    # 3. Test Bucket C (Expect Low p-values, except for Raw Liq Prices)
    test_bucket_hypothesis(df_final, [a for a in  df_bucket_c.columns.to_list() if ("BAR_TIMESTAMP" not in a) and ("EXCHANGE" not in a) and ("SYMBOL" not in a)], "BUCKET_C")


    return df_final

# ==========================================
# MAIN EXECUTION
# ==========================================
DEFAULT_FILE_PATH = "./sample_data/btc_5min_nan_processed_data.csv"
RUN_FILE = "./scripts/run_analysis_refactor_p2.py"
DAYS_PER_MONTH = 30
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
MINUTES_PER_ROW = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True, help="Output directory root")
    parser.add_argument("--num_splits", type=int, default= -1,  help="Number of splits")
    parser.add_argument("--num_folds", type=int, default= 3,  help="Number of folds")
    parser.add_argument("--filepath", type=str, default=DEFAULT_FILE_PATH, help="File path")
    parser.add_argument("--target_path", default="./sample_data/btc_5min_targets.csv", type=str, help="Path to target CSV file")
    parser.add_argument("--latest_num_months", default=-1, type=int, help="Number of months to use for latest bucket")
    parser.add_argument("--purge_size", default=24, type=int, help="Number of rows to purge during validation")

    args = parser.parse_args()
    
    output_folder = args.output_folder
    num_splits = args.num_splits    
    FILE_PATH = args.filepath
    num_months = args.latest_num_months
    purge_size = args.purge_size
    num_folds= args.num_folds

    # 1. Setup Main Directory and Logging
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- NEW: PREPARE ENVIRONMENT TO SILENCE WARNINGS ---
    # This creates a copy of your OS environment and forces Python 
    # to ignore warnings in the subprocess.
    my_env = os.environ.copy()
    my_env["PYTHONWARNINGS"] = "ignore"
    # ----------------------------------------------------
    
    log_file_path = os.path.join(output_folder, "log.txt")
    sys.stdout = Logger(log_file_path)

    try:
        # 2. Load Data
        print("Loading Data...")
        
        df = pd.read_csv(FILE_PATH, parse_dates=[TIME_STAMP_NAME])
        df.sort_values([TIME_STAMP_NAME], inplace=True)
        
        # 3. Update Bucket C List (Dynamic handling for liq prices)
        bucket_c_cols = [col for col in bucket_c_cols if col not in liq_transform_map.keys()]
        bucket_c_cols.extend(liq_transform_map.values())
        print(f"Updated Bucket C List: {bucket_c_cols}")


        # 4. Run Hypothesis Pipeline
        test_bucket_hypothesis(df, bucket_a_cols, "Bucket A")
        test_bucket_hypothesis(df, bucket_b_cols, "Bucket B")     
        test_bucket_hypothesis(df, bucket_c_cols, "Bucket C")



        # 5. Generate Splits
        if num_months == -1 :
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

                # C. Define Directory Structure: output_folder/split_N/data_split/
                split_dir = os.path.join(output_folder, f"split_{split_id}")
                data_split_dir = os.path.join(split_dir, "data_split")
                
                if not os.path.exists(data_split_dir):
                    os.makedirs(data_split_dir)

                # D. Save Processed Split
                global_start = split[0]
                global_end = split[-1]
                X_path = os.path.join(data_split_dir, f"X_{global_start}_{global_end}.csv")
                
                print(f"Saving split data to: {X_path}")
                X.to_csv(X_path, index=False)

                # --- MEMORY CLEANUP BEFORE SUBPROCESS ---
                print("Cleaning up memory before subprocess...")
                del X
                del df_split
                gc.collect()
                # ----------------------------------------

                # E. Run Analysis Subprocess
                print(f"Launching subprocess: {RUN_FILE}")
                print(f"purge_size = {purge_size}")

                # Use Popen to stream output line-by-line
                with subprocess.Popen(
                    [
                        sys.executable,
                        RUN_FILE,
                        "--X_path", X_path,
                        "--split_dir", split_dir,
                        "--target_path", args.target_path,
                        "--purge_size", str(purge_size),
                        "--num_folds", str(num_folds)
                    ],
                    stdout=subprocess.PIPE,  # Capture stdout
                    stderr=subprocess.STDOUT, # Redirect stderr to stdout so errors define logged too
                    text=True,               # Decode to string immediately
                    bufsize=1,           # Line buffering
                    env=my_env  # <--- CRITICAL FIX: Pass the modified environment
                ) as proc:
                    # Read output line by line as it is generated
                    for line in proc.stdout:
                        print(line, end='')  # This triggers your Logger.write() (Terminal + File)

                if proc.returncode != 0:
                    print(f"Subprocess failed with return code {proc.returncode}")
                    raise subprocess.CalledProcessError(proc.returncode, RUN_FILE)
        else:
            # 1. Calculate number of rows to process
            rows_to_process = int(num_months * DAYS_PER_MONTH * HOURS_PER_DAY * MINUTES_PER_HOUR / MINUTES_PER_ROW)
            
            # Safety check: ensure we don't try to grab more rows than exist
            if rows_to_process > len(df):
                print(f"Warning: Requested {num_months} months ({rows_to_process} rows) exceeds data length. Using full dataset.")
                rows_to_process = len(df)

            # 2. Generate the indices for the LAST 'rows_to_process'
            start_index = len(df) - rows_to_process
            split = np.arange(start_index, len(df))

            # Set up identification for this single run
            split_id = 1 
            print(f"\n{'='*40}")
            print(f"PROCESSING LAST {num_months} MONTHS ({rows_to_process} rows)")
            print(f"{'='*40}")

            # --- LOGIC COPIED FROM SPLIT LOOP BELOW ---

            # Create copy to avoid SettingWithCopy warnings
            df_split = df.iloc[split].copy()
            
            # A. Inplace Transformation (Liquidation columns)
            transform_liq_prices_update_inplace(df_split, verbose=True)

            # B. Run Pipeline
            X = run_X_pipeline(df_split, verbose=True)

            # C. Define Directory Structure: output_folder/split_recent/data_split/
            split_dir = os.path.join(output_folder, f"split_recent_{num_months}m")
            data_split_dir = os.path.join(split_dir, "data_split")
            
            if not os.path.exists(data_split_dir):
                os.makedirs(data_split_dir)

            # D. Save Processed Split
            global_start = split[0]
            global_end = split[-1]
            X_path = os.path.join(data_split_dir, f"X_{global_start}_{global_end}.csv")
            
            print(f"Saving split data to: {X_path}")
            X.to_csv(X_path, index=False)

            # --- MEMORY CLEANUP BEFORE SUBPROCESS ---
            print("Cleaning up memory before subprocess...")
            del X
            del df_split
            gc.collect()
            # ----------------------------------------

            # E. Run Analysis Subprocess

            print(f"Launching subprocess: {RUN_FILE}")
            print(f"purge_size = {purge_size}")
            # Use Popen to stream output line-by-line
            with subprocess.Popen(
                [
                    sys.executable,
                    RUN_FILE,
                    "--X_path", X_path,
                    "--split_dir", split_dir,
                    "--target_path", args.target_path, 
                    "--num_folds", str(num_folds) ,
                    "--purge_size", str(purge_size)
                ],
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,               # Decode to string immediately
                bufsize=1,                 # Line buffering
                env=my_env  # <--- CRITICAL FIX: Pass the modified environment
            ) as proc:
                # Read output line by line as it is generated
                for line in proc.stdout:
                    print(line, end='')  # This triggers your Logger.write()

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