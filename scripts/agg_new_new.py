import argparse
import re
import ast
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from scipy.stats import hmean
import numpy as np

# -------------------------------------------------------------------------
# 1. ROBUST PARSING LOGIC
# -------------------------------------------------------------------------

def parse_log_file_robust(filepath):
    cluster_map = {}
    importance_data = []
    importance_headers = []
    n_folds = 5 # Default fallback
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- CAPTURE N_FOLDS ---
        # Look for: "Starting 5-Fold Walk-Forward MDA"
        if "Fold Walk-Forward MDA" in line:
            match = re.search(r"Starting\s+(\d+)-Fold", line)
            if match:
                n_folds = int(match.group(1))

        # --- STATE 1: CLUSTER CONFIG ---
        if "--- FINAL CLUSTER CONFIGURATION" in line:
            i += 1
            while i < len(lines) and "--- ALIGNING" not in lines[i]:
                line = lines[i].strip()
                if line.startswith("Cluster"):
                    # Matches "Cluster 1 (3 features):"
                    cluster_match = re.match(r"Cluster\s+(\d+)\s+\(\d+\s+features\):", line)
                    if cluster_match:
                        c_id = int(cluster_match.group(1))
                        # Look ahead for the feature list
                        if i + 1 < len(lines):
                            feature_line = lines[i+1].strip()
                            try:
                                features = ast.literal_eval(feature_line)
                                cluster_map[c_id] = sorted(features)
                                i += 1 
                            except: pass
                i += 1
            continue

        # --- STATE 2: IMPORTANCE TABLE (Sharpe Ratios) ---
        if "--- FINAL CLUSTER IMPORTANCE" in line:
            header_candidates = []
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                
                if "--- DECISION" in line or "ERROR" in line or (line == "" and len(importance_data) > 0):
                    if len(importance_data) > 0 and line == "": 
                        pass 
                    elif "---" in line or "ERROR" in line: 
                        break
                
                if line and line[0].isdigit():
                    parts = line.split()
                    row_dict = {}
                    
                    if len(importance_headers) > 0:
                        if len(parts) == len(importance_headers) + 1:
                            row_dict['Cluster_ID'] = int(parts[0])
                            for h, val in zip(importance_headers, parts[1:]):
                                try: row_dict[h] = float(val)
                                except: pass
                            importance_data.append(row_dict)
                        elif len(parts) == len(importance_headers):
                            row_dict['Cluster_ID'] = int(parts[0])
                            for h, val in zip(importance_headers[1:], parts[1:]):
                                try: row_dict[h] = float(val)
                                except: pass
                            importance_data.append(row_dict)

                    i += 1
                    continue

                if "Cluster_ID" in line:
                    current_parts = line.split()
                    if len(current_parts) == 1 and len(header_candidates) > 0:
                        importance_headers = header_candidates[-1].split()
                    else:
                        importance_headers = current_parts[1:] 
                else:
                    if line: header_candidates.append(line)
                i += 1
            continue
        i += 1

    if not cluster_map or not importance_data:
        print(f"Error: Incomplete data in {filepath}.")
        sys.exit(1)

    df = pd.DataFrame(importance_data)
    if 'Cluster_ID' in df.columns:
        df.set_index('Cluster_ID', inplace=True)
    
    return cluster_map, df, n_folds

# -------------------------------------------------------------------------
# 2. ALIGNMENT LOGIC
# -------------------------------------------------------------------------

def get_cluster_signature(feature_list):
    return tuple(sorted(feature_list))

def align_runs(file_paths, custom_names=None):
    master_path = file_paths[0]
    print(f"Parsing Master Template: {os.path.basename(master_path)}...")
    master_map, master_df, master_n_folds = parse_log_file_robust(master_path)
    
    sig_to_master_id = {get_cluster_signature(feats): cid for cid, feats in master_map.items()}
    master_name = custom_names[0] if custom_names else os.path.basename(master_path)
    
    aligned_dfs = {}
    aligned_dfs[master_name] = master_df
    all_targets = set(master_df.columns.tolist())

    for idx, fp in enumerate(file_paths[1:], 1):
        fname = custom_names[idx] if custom_names and idx < len(custom_names) else os.path.basename(fp)
        print(f"Processing: {fname}...")
        
        curr_map, curr_df, _ = parse_log_file_robust(fp)
        all_targets.update(curr_df.columns.tolist())

        curr_sig_to_id = {get_cluster_signature(feats): cid for cid, feats in curr_map.items()}
        
        # Validate Structure
        master_sigs = set(sig_to_master_id.keys())
        curr_sigs = set(curr_sig_to_id.keys())
        if master_sigs != curr_sigs:
            print(f"\nCRITICAL ERROR: Cluster mismatch in {fname}!")
            sys.exit(1)
            
        id_mapper = {}
        for sig, m_id in sig_to_master_id.items():
            c_id = curr_sig_to_id[sig]
            id_mapper[c_id] = m_id
            
        curr_df.rename(index=id_mapper, inplace=True)
        curr_df.sort_index(inplace=True)
        aligned_dfs[fname] = curr_df

    return aligned_dfs, master_map, sorted(list(all_targets)), master_n_folds

# -------------------------------------------------------------------------
# 3. PLOTTING LOGIC (DYNAMIC SHARPE THRESHOLDS)
# -------------------------------------------------------------------------

def create_y_labels(cluster_map):
    labels = []
    sorted_ids = sorted(cluster_map.keys())
    for cid in sorted_ids:
        features = cluster_map[cid]
        feat_str = ", ".join(features)
        wrapped_feat = textwrap.fill(feat_str, width=60)
        labels.append(f"ID {cid}\n[{wrapped_feat}]")
    return labels

def calculate_rank_stats(data_matrix, run_names):
    """
    Higher Sharpe = Better Rank.
    """
    runs_df = data_matrix[run_names].copy()
    # Ascending=True -> Smallest number gets Rank 1 (Worst)
    ranks_df = runs_df.rank(axis=0, method='average', ascending=True)
    
    # Harmonic Mean (Robust)
    hm_series = ranks_df.apply(lambda row: hmean(row) if (row > 0).all() else 0, axis=1)
    # Arithmetic Mean
    mean_series = ranks_df.mean(axis=1)
    
    return hm_series, mean_series

def plot_advanced_heatmaps(aligned_dfs, cluster_map, targets, n_folds):
    y_labels = create_y_labels(cluster_map)
    run_names = list(aligned_dfs.keys()) 
    plot_height = max(8, len(cluster_map) * 1.0) 

    # --- DYNAMIC THRESHOLD CALCULATION ---
    # T-Stat approx = Sharpe * sqrt(N)
    # For 95% Confidence (t ~ 2.0): Threshold = 2.0 / sqrt(N)
    sig_threshold = 2.0 / np.sqrt(n_folds)
    
    print(f"\n--- VISUALIZATION SETTINGS ---")
    print(f"Detected N_FOLDS: {n_folds}")
    print(f"Significance Threshold (T): {sig_threshold:.4f}")
    print(f"Heatmap Center: {sig_threshold:.4f} (Green > T, Red < T)")

    def plot_single_target(data_matrix, title_text):
        width_ratios = [len(run_names) + 1, 2]
        
        fig, axes = plt.subplots(1, 2, figsize=(24, plot_height), 
                                 gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05},
                                 constrained_layout=True)
        
        # --- LEFT PLOT: SHARPE SCORES ---
        cols_scores = run_names + ['TOTAL_SCORE']
        df_plot_scores = data_matrix[cols_scores]
        
        # Center the map at the Significance Threshold
        # Red = Noise/Harmful (Below Threshold)
        # Green = Signal (Above Threshold)
        
        # Determine range to make the visual balanced
        # We want the transition to be at sig_threshold.
        # So we shift the data so that sig_threshold becomes 0 for the cmap, 
        # or we just trust 'center' arg. 
        # Note: 'center' in sns.heatmap fixes the 'white' color at that value.
        
        sns.heatmap(df_plot_scores, annot=True, fmt=".2f", 
                    cmap="RdYlGn", 
                    center=sig_threshold, # Dynamic Center!
                    vmin=-2.0,  # Wider floor (makes red less aggressive for small negatives)
                    vmax=5.0,   # Wider ceiling (makes green gradual up to 5.0)
                    yticklabels=y_labels, ax=axes[0], 
                    cbar_kws={'label': f'Sharpe Ratio (Center={sig_threshold:.2f})'})
        
        axes[0].set_title(f"{title_text}\n(Metric: Sharpe Ratio | Threshold T={sig_threshold:.2f})", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Cluster ID & Features")
        axes[0].set_xlabel("Inputs")

        # --- RIGHT PLOT: RANK METRICS ---
        cols_rank = ['RANK_HM', 'RANK_MEAN']
        df_plot_ranks = data_matrix[cols_rank]
        
        sns.heatmap(df_plot_ranks, annot=True, fmt=".2f", cmap="Oranges", 
                    yticklabels=False, ax=axes[1], cbar_kws={'label': 'Rank (Higher is Better)'})
        
        axes[1].set_title("Rank Metrics", fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(['Harmonic\nMean', 'Arithmetic\nMean'], rotation=0)
        axes[1].set_xlabel("Aggregations")
        
        plt.show()

    # --- PART A: Individual Target Plots ---
    for target in targets:
        print(f"Generating Plot for Target: {target}")
        
        data_matrix = pd.DataFrame()
        for run_name in run_names:
            df = aligned_dfs[run_name]
            if target in df.columns:
                data_matrix[run_name] = df[target]
            else:
                data_matrix[run_name] = 0.0 
        
        # MEAN Sharpe (Not Sum)
        data_matrix['TOTAL_SCORE'] = data_matrix[run_names].mean(axis=1)
        
        hm, mean = calculate_rank_stats(data_matrix, run_names)
        data_matrix['RANK_HM'] = hm
        data_matrix['RANK_MEAN'] = mean
        
        data_matrix.sort_index(inplace=True)
        
        plot_single_target(data_matrix, f"Target: {target}")

    # --- PART B: Global Summary Plot ---
    print("Generating Global Summary Plot...")
    
    sum_matrix = pd.DataFrame()
    for run_name in run_names:
        df = aligned_dfs[run_name]
        valid_cols = [c for c in df.columns if c in targets]
        # Average Sharpe across targets for this run
        sum_matrix[run_name] = df[valid_cols].mean(axis=1)
    
    # Grand Mean Sharpe
    sum_matrix['TOTAL_SCORE'] = sum_matrix[run_names].mean(axis=1)
    
    hm, mean = calculate_rank_stats(sum_matrix, run_names)
    sum_matrix['RANK_HM'] = hm
    sum_matrix['RANK_MEAN'] = mean
    
    sum_matrix.sort_index(inplace=True)
    
    plot_single_target(sum_matrix, "COMBINED SUMMARY (Mean of Sharpe Ratios)")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Cluster Importance (Sharpe Based)")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='List of log file paths')
    parser.add_argument('--names', type=str, nargs='+', help='Custom column headers for the log files')
    
    args = parser.parse_args()
    
    if len(args.files) < 1:
        print("Please provide at least one log file.")
        sys.exit(1)
        
    if args.names and len(args.names) != len(args.files):
        print(f"Error: {len(args.files)} files provided but {len(args.names)} names provided.")
        sys.exit(1)
        
    aligned_dfs, master_map, targets, n_folds = align_runs(args.files, args.names)
    plot_advanced_heatmaps(aligned_dfs, master_map, targets, n_folds)