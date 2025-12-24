import pandas as pd
import numpy as np
from numba import njit
import os
import argparse
import sys

# ==========================================
# 1. THE NUMBA ENGINE (Triple Barrier)
# ==========================================
@njit
def _triple_barrier_kernel(price_values, vol_values, width, expiry):
    n = len(price_values)
    out = np.zeros(n, dtype=np.int8) 

    for t in range(n - expiry):
        v_t = vol_values[t]
        if np.isnan(v_t) or v_t == 0: continue
            
        p_t = price_values[t]
        # Barrier distance based on volatility
        barrier_dist = p_t * v_t * width
        upper = p_t + barrier_dist
        lower = p_t - barrier_dist
        
        for i in range(1, expiry + 1):
            p_future = price_values[t + i]
            hit_up = p_future >= upper
            hit_down = p_future <= lower
            
            if hit_up and hit_down:
                out[t] = -1 # Touched both (whipsaw), mark as fail or side
                break
            elif hit_up:
                out[t] = 1
                break
            elif hit_down:
                out[t] = -1
                break
    return out

# ==========================================
# 2. WRAPPER FUNCTIONS
# ==========================================
def get_triple_barrier_label(price_series, vol_series, width, expiry):
    """
    Standard Triple Barrier Method (Fixed Width * Volatility).
    """
    p_arr = price_series.to_numpy().astype(np.float64)
    v_arr = vol_series.to_numpy().astype(np.float64)
    
    # Scale Volatility: Scale 1-period vol by sqrt(expiry)
    v_scaled = v_arr * np.sqrt(expiry)
    
    # Pass the SCALED volatility to the kernel
    labels_arr = _triple_barrier_kernel(p_arr, v_scaled, width, expiry)
    return pd.Series(labels_arr, index=price_series.index)

def get_dynamic_buy_hold_label(price_series, vol_series, expiry, mode, sigma_mult=0.5):
    """
    Dynamic Buy/Hold Label.
    Target = Current Vol * sqrt(Time) * Sigma_Multiplier
    """
    p_arr = price_series.to_numpy().astype(np.float64)
    v_arr = vol_series.to_numpy().astype(np.float64)
    
    # Calculate future price (vectorized)
    future_price = np.full_like(p_arr, np.nan)
    future_price[:-expiry] = p_arr[expiry:]
    
    # Calculate Return
    with np.errstate(divide='ignore', invalid='ignore'):
        ret = (future_price - p_arr) / p_arr
    ret = np.nan_to_num(ret, nan=0.0)
    
    # Calculate Dynamic Threshold
    period_vol = v_arr * np.sqrt(expiry)
    dynamic_threshold = period_vol * sigma_mult
    
    # Generate Labels
    if mode == 'long':
        labels = np.where(ret > dynamic_threshold, 1, 0)
    elif mode == 'short':
        labels = np.where(ret < -dynamic_threshold, 1, 0)
    else:
        labels = np.zeros(len(p_arr), dtype=int)
        
    return pd.Series(labels, index=price_series.index)

# ==========================================
# 3. GENERATE TARGETS & STATISTICS
# ==========================================
def generate_targets(df, params, price_col='CLOSE_PRICE'):
    print(f"--- Processing Data ---")
    print(f"Hyperparameters: {params}")
    
    # Unpack params
    expiry = params['expiry']
    vol_span = params['vol_lookback']
    tb_width = params['tb_width']
    dyn_mult = params['dyn_mult'] # Same multiplier for Long and Short

    # Ensure we are working with a copy
    cols_to_keep = ['EXCHANGE', 'SYMBOL', 'BAR_TIMESTAMP', price_col]
    work_df = df[cols_to_keep].copy()
    
    # 1. Pre-process logic (Smoothing)
    work_df['smooth_price'] = work_df[price_col].ewm(span=5).mean()
    
    # 2. Calculate Volatility (EWMA Standard Deviation)
    print(f"Calculating Volatility (Span={vol_span})...")
    rets = work_df['smooth_price'].pct_change()
    vol_long = rets.ewm(span=vol_span).std().fillna(0)

    print("--- Generating Targets ---")

    # Target A: Triple Barrier
    col_tb = f'target_tb_{tb_width}_{expiry}'
    print(f"Computing {col_tb}...")
    target_tb = get_triple_barrier_label(
        work_df['smooth_price'], vol_long, width=tb_width, expiry=expiry
    )

    # Target B: Dynamic Long
    col_dyn_long = f'target_long_dyn_{dyn_mult}_{expiry}'
    print(f"Computing {col_dyn_long}...")
    target_dyn_long = get_dynamic_buy_hold_label(
        work_df['smooth_price'], vol_long, expiry=expiry, mode='long', sigma_mult=dyn_mult
    )
    
    # Target C: Dynamic Short
    col_dyn_short = f'target_short_dyn_{dyn_mult}_{expiry}'
    print(f"Computing {col_dyn_short}...")
    target_dyn_short = get_dynamic_buy_hold_label(
        work_df['smooth_price'], vol_long, expiry=expiry, mode='short', sigma_mult=dyn_mult
    )

    # 3. CONSTRUCT FINAL DF
    targets_df = pd.DataFrame(index=work_df.index)
    targets_df['EXCHANGE'] = work_df['EXCHANGE']
    targets_df['SYMBOL'] = work_df['SYMBOL']
    targets_df['BAR_TIMESTAMP'] = work_df['BAR_TIMESTAMP']
    
    targets_df[col_tb] = target_tb
    targets_df[col_dyn_long] = target_dyn_long
    targets_df[col_dyn_short] = target_dyn_short
    
    # 4. PRINT TRIGGER STATISTICS
    print("\n" + "="*40)
    print(" TARGET DISTRIBUTION STATISTICS")
    print("="*40)
    
    target_cols = [col_tb, col_dyn_long, col_dyn_short]
    
    for col in target_cols:
        counts = targets_df[col].value_counts(dropna=False).sort_index()
        total = len(targets_df)
        
        print(f"\nTarget: {col}")
        print("-" * 30)
        print(f"{'Label':<10} | {'Count':<10} | {'Freq (%)':<10}")
        print("-" * 30)
        
        for label, count in counts.items():
            freq = (count / total) * 100
            print(f"{str(label):<10} | {count:<10} | {freq:.2f}%")

    print("\n" + "="*40)
    
    return targets_df

# ==========================================
# 4. ARGUMENT PARSING & EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Triple Barrier and Dynamic Buy/Hold Targets.')

    # --- File Paths ---
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input CSV file (processed data)')
    parser.add_argument('--out_dir', type=str, default="./processed_data",
                        help='Directory to save the output file')
    parser.add_argument('--out_name', type=str, default="btc_5min_targets.csv",
                        help='Name of the output CSV file')

    # --- Hyperparameters ---
    parser.add_argument('--expiry', type=int, default=24,
                        help='Lookahead window (number of bars). Default: 24')
    parser.add_argument('--vol_lookback', type=int, default=100,
                        help='Span for EWM volatility calculation. Default: 100')
    parser.add_argument('--tb_width', type=float, default=1.5,
                        help='Multiplier for Triple Barrier width. Default: 1.5')
    parser.add_argument('--dyn_mult', type=float, default=1.0,
                        help='Multiplier for Dynamic Long/Short targets. Default: 1.0')

    args = parser.parse_args()

    # Consolidate params for the function
    params = {
        "vol_lookback": args.vol_lookback,
        "expiry": args.expiry,
        "tb_width": args.tb_width,
        "dyn_mult": args.dyn_mult 
    }

    print(f"Reading file: {args.input}")
    
    if os.path.exists(args.input):
        # Load Data
        df = pd.read_csv(args.input, parse_dates=["BAR_TIMESTAMP"])
        
        # Run Generation
        new_targets = generate_targets(df, params, price_col='CLOSE_PRICE')
        
        # Save output
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        output_full_path = os.path.join(args.out_dir, args.out_name)
        new_targets.to_csv(output_full_path, index=False)
        print(f"Saved targets to: {output_full_path}")
    else:
        print(f"Error: Input file not found at {args.input}")
        sys.exit(1)