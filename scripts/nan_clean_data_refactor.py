import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logger(log_dir, log_filename="pipeline.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("DataPipeline")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_path, mode='w')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CORE_PRICE_COLS = [
    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',       
    'VWAP', 'AVG_PRICE',                                            
    'MARK_PRICE_LAST', 'INDEX_PRICE_LAST', 'LAST_PRICE_LAST',     
    'OPEN_INTEREST_LAST', 'FUNDING_RATE_LAST'  
]

CORE_VOL_COLS = [
    'VOLUME', 'BUY_VOLUME', 'SELL_VOLUME', 
    'NOTIONAL_VOLUME', 'BUY_NOTIONAL', 'SELL_NOTIONAL',
    'TRADES_COUNT', 'TOTAL_TRADES_COUNT', 'BUY_TRADES_COUNT', 'SELL_TRADES_COUNT',
    'TICK_COUNT',      
    'PRICE_STDDEV', 'TRADE_SIZE_STDDEV',
    'AVG_TRADE_SIZE', 'MIN_TRADE_SIZE', 'MAX_TRADE_SIZE',
    'VOLUME_IMBALANCE', 'TRADE_COUNT_IMBALANCE'
]

LIQ_FLOW_COLS = [
    'BUY_LIQUIDATION_VOLUME', 'SELL_LIQUIDATION_VOLUME', 'TOTAL_LIQUIDATION_VOLUME',
    'BUY_LIQUIDATION_COUNT', 'SELL_LIQUIDATION_COUNT', 'TOTAL_LIQUIDATION_COUNT',
    'BUY_LIQUIDATION_NOTIONAL', 'SELL_LIQUIDATION_NOTIONAL', 'TOTAL_LIQUIDATION_NOTIONAL',
    'AVG_BUY_LIQUIDATION_SIZE', 'AVG_SELL_LIQUIDATION_SIZE', 'AVG_LIQUIDATION_SIZE',
    'MAX_SINGLE_LIQUIDATION_SIZE', 'MAX_SINGLE_LIQUIDATION_NOTIONAL',
    'LIQUIDATION_VOLUME_IMBALANCE', 'LIQUIDATION_COUNT_IMBALANCE'
]

LIQ_PRICE_COLS = [
    'AVG_LIQUIDATION_PRICE', 'VWAP_LIQUIDATION_PRICE', 
    'MIN_LIQUIDATION_PRICE', 'MAX_LIQUIDATION_PRICE'
]

# ==============================================================================
# ANALYSIS HELPER FUNCTIONS
# ==============================================================================

def data_check_structured(df):
    results = {}
    
    # 1. NaN Analysis
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

    # 2. Timestamp Consistency
    delta_minutes = df["BAR_TIMESTAMP"].diff().dt.total_seconds() / 60
    missing_intervals = df[delta_minutes != 5].copy()
    if not missing_intervals.empty:
        missing_intervals["delta_minutes"] = delta_minutes[delta_minutes != 5]
    
    results['timestamp_summary'] = {
        "total_rows": len(df),
        "duplicate_timestamps": int(df['BAR_TIMESTAMP'].duplicated().sum()),
        "non_5min_count": len(missing_intervals),
        "non_5min_percent": round(100 * len(missing_intervals) / len(df), 2)
    }
    
    # 3. Numeric Stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].agg(['min', 'max', 'mean', 'std']).T
        results['numeric_stats'] = stats.round(4)
    else:
        results['numeric_stats'] = pd.DataFrame()

    return results

def log_full_data_check(results, stage_name):
    """Dumps the full dictionary content of data_check_structured to the logger."""
    logger.info(f"\n{'='*30}")
    logger.info(f"DATA CHECK DUMP: {stage_name}")
    logger.info(f"{'='*30}")

    # 1. Timestamp Summary
    logger.info("--- [1] TIMESTAMP SUMMARY ---")
    for k, v in results['timestamp_summary'].items():
        logger.info(f"   {k}: {v}")

    # 2. NaN Analysis (Full Dump or Top Significant)
    logger.info("\n--- [2] NAN ANALYSIS (Top 15 Columns by Count) ---")
    if not results['nan_analysis'].empty:
        df_nan = results['nan_analysis'].sort_values("nan_count", ascending=False)
        logger.info(f"\n{df_nan.head(15).to_string(index=False)}")
    else:
        logger.info("   No NaNs found.")

    # 3. Numeric Statistics
    logger.info("\n--- [3] NUMERIC STATISTICS (Sample) ---")
    if not results['numeric_stats'].empty:
        logger.info(f"\n{results['numeric_stats'].head(10).to_string()}")
    else:
        logger.info("   No numeric columns found.")
    
    logger.info(f"{'='*30}\n")

def plot_data(df, name_prefix, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_df = df.copy()
    plot_df['BAR_TIMESTAMP'] = pd.to_datetime(plot_df['BAR_TIMESTAMP'], errors='coerce')
    plot_df = plot_df.sort_values("BAR_TIMESTAMP").reset_index(drop=True)

    logger.info(f"   >> Generating plots for: {name_prefix}")

    # 1. NaNs per row over time
    nan_counts = plot_df.isna().sum(axis=1)
    rolling_window = 50 
    nan_counts_smoothed = nan_counts.rolling(rolling_window).mean()
    
    plt.figure(figsize=(15,4))
    plt.plot(plot_df["BAR_TIMESTAMP"], nan_counts_smoothed)
    plt.title(f"{name_prefix}: Smoothed NaNs per row (window={rolling_window})")
    plt.xlabel("Timestamp")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{name_prefix}_nan_temporal.png"))
    plt.close()

    # 2. Timestamp gaps
    plot_df["delta_minutes"] = plot_df["BAR_TIMESTAMP"].diff().dt.total_seconds() / 60
    plt.figure(figsize=(15,4))
    plt.plot(plot_df["BAR_TIMESTAMP"], plot_df["delta_minutes"], marker='o', linestyle='None', markersize=2)
    plt.title(f"{name_prefix}: Time Gaps (minutes)")
    if plot_df["delta_minutes"].max() > 0:
        plt.ylim(0, plot_df["delta_minutes"].max() * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{name_prefix}_time_gaps.png"))
    plt.close()

# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def log_step(step_name, df, extra_msg=""):
    logger.info(f"\n[STEP: {step_name}]")
    logger.info(f"   Shape: {df.shape}")
    if extra_msg:
        logger.info(f"   Info: {extra_msg}")

def load_and_prep_source(filepath, name, plot_folder):
    logger.info(f"\n{'='*40}")
    logger.info(f"LOADING DATASET: {name}")
    logger.info(f"{'='*40}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["BAR_TIMESTAMP"]).sort_values("BAR_TIMESTAMP").reset_index(drop=True)
    logger.info(f"   Initial Shape: {df.shape}")
    
    # Specific logic for Liquidations
    if name == 'liquidations':
        logger.info("   >> Filling NaNs with 0 for Liquidations")
        df.fillna(0, inplace=True)
        # ASSERTION: Ensure liquidations are clean immediately
        assert df.isna().sum().sum() == 0, "Initial Liquidations DF still has NaNs!"

    # --- LOG DUMP: INITIAL ---
    results = data_check_structured(df)
    log_full_data_check(results, f"INITIAL LOAD - {name}")

    plot_data(df, name, plot_folder)
    return df

def merge_datasets(dfs):
    logger.info(f"\n{'='*40}")
    logger.info(f"MERGING {len(dfs)} DATASETS")
    logger.info(f"{'='*40}")
    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="BAR_TIMESTAMP", how="outer"), dfs)
    df_merged = df_merged.sort_values("BAR_TIMESTAMP").reset_index(drop=True)
    
    log_step("Merge Complete", df_merged, f"Initial NaNs: {df_merged.isna().sum().sum()}")
    return df_merged

def reindex_and_fill_missing_timestamps(df):
    """
    Creates a complete 5-minute index from min to max timestamp.
    Fills missing rows with NaNs.
    """
    logger.info(f"\n[STEP: Filling Missing Time Intervals]")
    
    min_time = df['BAR_TIMESTAMP'].min()
    max_time = df['BAR_TIMESTAMP'].max()
    full_idx = pd.date_range(start=min_time, end=max_time, freq='5min', name='BAR_TIMESTAMP')
    
    logger.info(f"   Time Range: {min_time} to {max_time}")
    logger.info(f"   Expected Rows: {len(full_idx)}")
    
    df = df.set_index('BAR_TIMESTAMP')
    if df.index.duplicated().any():
        logger.warning(f"   Found {df.index.duplicated().sum()} duplicate timestamps. Keeping first.")
        df = df[~df.index.duplicated(keep='first')]

    df_reindexed = df.reindex(full_idx)
    df_reindexed = df_reindexed.reset_index() 
    
    added_rows = len(df_reindexed) - len(df)
    log_step("Timestamp Filling", df_reindexed, f"Added {added_rows} rows of NaNs.")
    
    # ASSERTION: Ensure exact length match
    assert len(df_reindexed) == len(full_idx), "Reindexed DF length does not match expected time range!"
    
    return df_reindexed

def clean_merge_artifacts(df):
    cols_to_drop = [
        'EXCHANGE', 'SYMBOL', 'EXCHANGE_y', 'SYMBOL_y', 'EXCHANGE.1', 'SYMBOL.1',
        'delta_minutes_y', 'delta_minutes_x', 'delta_minutes'
    ]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    
    rename_map = {'EXCHANGE_x': 'EXCHANGE', 'SYMBOL_x': 'SYMBOL'}
    df.rename(columns=rename_map, inplace=True)

    # --- CRITICAL FIX: Propagate Metadata into Gaps ---
    metadata_cols = ['EXCHANGE', 'SYMBOL']
    for col in metadata_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill() 
            
            # ASSERTION: Metadata must be clean
            nans_meta = df[col].isna().sum()
            assert nans_meta == 0, f"Column {col} still has {nans_meta} NaNs after fix!"

    log_step("Artifact Cleanup", df, f"Dropped {len(existing_drop)} cols & Filled Metadata.")
    return df

def clean_market_data(df):
    # A. Prices -> Forward Fill (Physical Reality: Last price holds)
    existing_price_cols = [c for c in CORE_PRICE_COLS if c in df.columns]
    df[existing_price_cols] = df[existing_price_cols].ffill()
    
    # B. Volumes -> Fill 0 (Physical Reality: No trade = 0 volume)
    existing_vol_cols = [c for c in CORE_VOL_COLS if c in df.columns]
    df[existing_vol_cols] = df[existing_vol_cols].fillna(0)
    
    # ASSERTION: Volumes must be 100% clean
    vol_nans = df[existing_vol_cols].isna().sum().sum()
    assert vol_nans == 0, f"Volume columns still have {vol_nans} NaNs!"
    
    log_step("Market Data Cleaning", df, "FFilled Prices, 0-Filled Volumes.")
    return df, existing_price_cols

def clean_liquidation_data(df):
    # A. Flows -> Fill 0
    existing_liq_flow = [c for c in LIQ_FLOW_COLS if c in df.columns]
    df[existing_liq_flow] = df[existing_liq_flow].fillna(0)

    # B. Prices -> Forward Fill
    existing_liq_price = [c for c in LIQ_PRICE_COLS if c in df.columns]
    df[existing_liq_price] = df[existing_liq_price].ffill()
    
    log_step("Liquidation Data Cleaning", df, "0-Filled Flows, FFilled Levels.")
    return df

def perform_safety_checks(df, price_cols):
    before_len = len(df)
    
    # Drop rows where we couldn't forward fill (start of dataset)
    df.dropna(subset=price_cols, inplace=True)
    dropped_count = before_len - len(df)
    
    log_step("Safety Check", df, f"Dropped {dropped_count} rows due to missing initial prices.")
    
    # ASSERTION: Critical prices must be clean
    price_nans = df[price_cols].isna().sum().sum()
    assert price_nans == 0, f"Critical Price columns still contain {price_nans} NaNs."
    
    return df

def engineer_timestamps(df):
    ts_cols = [c for c in df.columns if "TIMESTAMP" in c]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # Generate Features (Correctly handles gaps because gaps are now filled with ffill/0)
    df['LIQ_DURATION'] = (df['LAST_LIQUIDATION_TIMESTAMP'] - df['FIRST_LIQUIDATION_TIMESTAMP']).dt.total_seconds().fillna(0)
    df['LIQ_LATENCY'] = (df['LAST_LIQUIDATION_TIMESTAMP'] - df['BAR_TIMESTAMP']).dt.total_seconds().fillna(0)
    df['TRADE_SPAN'] = (df['LAST_TRADE_TIMESTAMP'] - df['FIRST_TRADE_TIMESTAMP']).dt.total_seconds().fillna(0)
    df['TICK_SPAN'] = (df['LAST_TICK_TIMESTAMP'] - df['FIRST_TICK_TIMESTAMP']).dt.total_seconds().fillna(0)
    df['SECONDS_SINCE_FUNDING'] = (df['BAR_TIMESTAMP'] - df['FUNDING_TIMESTAMP_LAST']).dt.total_seconds().fillna(0)
    
    # Drop Raw Metadata (Now safe to drop)
    ts_cols_to_drop = [
        'FIRST_LIQUIDATION_TIMESTAMP', 'LAST_LIQUIDATION_TIMESTAMP',
        'FIRST_TRADE_TIMESTAMP', 'LAST_TRADE_TIMESTAMP',
        'FIRST_TICK_TIMESTAMP', 'LAST_TICK_TIMESTAMP',
        'FUNDING_TIMESTAMP_LAST' 
    ]
    df.drop(columns=[c for c in ts_cols_to_drop if c in df.columns], inplace=True)
    
    log_step("Timestamp Engineering", df, "Generated Span features, dropped raw timestamps.")
    return df

def final_edge_case_cleanup(df):
    target_cols = [c for c in LIQ_PRICE_COLS if c in df.columns]
    if target_cols:
        before_len = len(df)
        if df[target_cols].isna().sum().sum() > 0:
            df.dropna(subset=target_cols, inplace=True)
            dropped = before_len - len(df)
            log_step("Final Edge Case Cleanup", df, f"Dropped {dropped} rows based on Liq Prices.")
    return df

def run_processing_pipeline(df_liq, df_deriv, df_trades, plot_folder):
    # 1. Merge
    dfs = [df_deriv, df_trades, df_liq]
    df_merged = merge_datasets(dfs)

    # 2. FILL MISSING TIMESTAMPS
    df_merged = reindex_and_fill_missing_timestamps(df_merged)

    # 3. Artifact Cleanup (WITH METADATA FIX)
    df_merged = clean_merge_artifacts(df_merged)

    # 4. Market Data Cleanup
    df_merged, price_cols = clean_market_data(df_merged)

    # 5. Liquidation Cleanup
    df_merged = clean_liquidation_data(df_merged)

    # 6. Safety Checks (Validates Step 4 & 5)
    df_merged = perform_safety_checks(df_merged, price_cols)

    # 7. Timestamp Engineering
    df_merged = engineer_timestamps(df_merged)

    # 8. Final Cleanup
    df_merged = final_edge_case_cleanup(df_merged)

    # 9. Final Stats & Plot
    final_nans = df_merged.isna().sum().sum()
    log_step("Final Dataset", df_merged, f"Final NaNs: {final_nans}")
    
    # --- FINAL ASSERTION: THE NUCLEAR OPTION ---
    if final_nans > 0:
        logger.critical(f"FATAL: Final dataset still has {final_nans} NaNs!")
        logger.critical(df_merged.columns[df_merged.isna().any()].tolist())
        raise AssertionError("Pipeline failed to produce clean data.")
    
    # --- LOG DUMP: FINAL ---
    final_results = data_check_structured(df_merged)
    log_full_data_check(final_results, "FINAL CLEANED DATASET")
    
    plot_data(df_merged, "final_clean", plot_folder)
    
    return df_merged

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    LIQ_PATH = "./sample_data/btc_5min_liquidations.csv"
    DERIV_PATH = "./sample_data/btc_5min_derivatives.csv"
    TRADES_PATH = "./sample_data/btc_5min_trades.csv"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--liq_path",   default=LIQ_PATH , type=str  )
    parser.add_argument("--deriv_path", default=DERIV_PATH , type=str )
    parser.add_argument("--trades_path",default=TRADES_PATH,  type=str )
    parser.add_argument("--output_dir", type=str, default="./processed_data")
    parser.add_argument("--output_filename", type=str, default="btc_5min_final_cleaned.csv")
    parser.add_argument("--plot_dir", type=str, default="./processed_data/pipeline_artifacts")
    args = parser.parse_args()

    logger = setup_logger(args.plot_dir)
    logger.info("Starting Data Pipeline...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        # Load and Plot Initial Data
        df_liq = load_and_prep_source(args.liq_path, "liquidations", args.plot_dir)
        df_deriv = load_and_prep_source(args.deriv_path, "derivatives", args.plot_dir)
        df_trades = load_and_prep_source(args.trades_path, "trades", args.plot_dir)

        # Run Pipeline
        df_final = run_processing_pipeline(df_liq, df_deriv, df_trades, args.plot_dir)
        
        # Save
        save_path = os.path.join(args.output_dir, args.output_filename)
        df_final.to_csv(save_path, index=False)
        logger.info(f"Saved final data to: {save_path}")

    except Exception as e:
        logger.critical(f"Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)