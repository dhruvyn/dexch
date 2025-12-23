import os
import re
import pandas as pd
import numpy as np
import argparse
import glob
import io
import ast
from scipy.special import softmax

def parse_log_file(filepath):
    """
    Extracts TWO things from the log:
    1. Cluster_ID -> [Feature List] mapping
    2. Cluster_ID -> Score table
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # --- STEP 1: EXTRACT CLUSTER DEFINITIONS ---
    cluster_map = {}
    
    # Regex to find "Cluster <id> (<count> features):\n<list>"
    cluster_def_pattern = re.compile(r"Cluster\s+(\d+)\s+\(\d+\s+features\):\s*\n(\[.*?\])")
    
    for match in cluster_def_pattern.finditer(content):
        c_id = int(match.group(1))
        feat_list_str = match.group(2)
        try:
            # Safely evaluate string representation of list "['a', 'b']" -> list
            feat_list = ast.literal_eval(feat_list_str)
            cluster_map[c_id] = feat_list
        except:
            continue

    if not cluster_map:
        return None

    # --- STEP 2: EXTRACT SCORES TABLE ---
    table_pattern = re.compile(
        r"--- FINAL CLUSTER IMPORTANCE \(Log Loss Decrease\) ---\n(.*?)\n(?:---|ERROR)", 
        re.DOTALL
    )
    match_table = table_pattern.search(content)

    if not match_table:
        return None

    raw_table = match_table.group(1).strip()
    
    try:
        # Parse table
        df_scores = pd.read_csv(io.StringIO(raw_table), sep=r'\s+', engine='python')
        
        # Handle index
        if 'Cluster_ID' in df_scores.columns:
            df_scores.set_index('Cluster_ID', inplace=True)
        elif df_scores.index.name != 'Cluster_ID':
            df_scores.index.name = 'Cluster_ID'
            
    except:
        return None

    # --- STEP 3: BROADCAST SCORES TO FEATURES ---
    feature_rows = []
    
    for c_id, scores in df_scores.iterrows():
        # Get all features belonging to this cluster ID
        features = cluster_map.get(c_id, [])
        
        for feat in features:
            # Assign the cluster's score to the individual feature
            row = scores.copy()
            row['Feature'] = feat
            feature_rows.append(row)
            
    if not feature_rows:
        return None
        
    df_features = pd.DataFrame(feature_rows)
    df_features.set_index('Feature', inplace=True)
    
    return df_features

def aggregate_features(root_dir, threshold):
    """
    Aggregates feature scores across all splits for a SPECIFIC THRESHOLD.
    Path: root_dir / run_* / split_* / analysis_runs / run_threshold_{t} / log.txt
    """
    # Specific Search Pattern for the given threshold
    search_pattern = os.path.join(
        root_dir, 
        "run_*", 
        "split_*", 
        "analysis_runs", 
        f"run_threshold_{threshold}", 
        "log.txt"
    )
    
    log_files = glob.glob(search_pattern, recursive=True)
    
    if not log_files:
        print(f"No log files found for threshold {threshold}")
        print(f"Checked pattern: {search_pattern}")
        return None

    print(f"Found {len(log_files)} logs for Threshold {threshold}. Parsing...")
    
    all_feature_dfs = []
    
    for f in log_files:
        df = parse_log_file(f)
        if df is not None:
            all_feature_dfs.append(df)
            
    if not all_feature_dfs:
        print("No valid data extracted.")
        return None
        
    # --- AGGREGATE ---
    print("Aggregating feature scores...")
    total_raw_scores = pd.concat(all_feature_dfs).groupby(level=0).sum()
    
    # Identify Columns
    cols = total_raw_scores.columns
    double_side = [c for c in cols if 'tb' in c.lower()]
    long_cols = [c for c in cols if 'long' in c.lower()]
    short_cols = [c for c in cols if 'short' in c.lower()]
    
    # Compute Softmax Ranks
    def get_rank(series):
        if series.sum() == 0: return 0.0
        return softmax(series.clip(lower=0))

    raw_sum_combined = total_raw_scores.sum(axis=1)
    
    summary_df = pd.DataFrame({
        'Feature': total_raw_scores.index,
        'Global_Rank_Score': get_rank(raw_sum_combined),
        'Long_Rank_Score': get_rank(total_raw_scores[long_cols].sum(axis=1)) if long_cols else 0,
        'Short_Rank_Score': get_rank(total_raw_scores[short_cols].sum(axis=1)) if short_cols else 0,
        'TripleBarrier_Rank_Score': get_rank(total_raw_scores[double_side].sum(axis=1)) if short_cols else 0,
        'Raw_LogLoss_Sum': raw_sum_combined
    })
    
    summary_df.set_index('Feature', inplace=True)
    summary_df.sort_values('Global_Rank_Score', ascending=False, inplace=True)
    
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing run folders")
    parser.add_argument("--t", type=float, required=True, help="Threshold to aggregate (e.g. 0.3)")
    args = parser.parse_args()
    
    df_result = aggregate_features(args.base_dir, args.t)
    
    if df_result is not None:
        filename = f"final_feature_ranking_t{args.t}.csv"
        output_path = os.path.join(args.base_dir, filename)
        
        df_result.to_csv(output_path)
        
        print("\n" + "="*60)
        print(f"FEATURE AGGREGATION COMPLETE (Threshold={args.t})")
        print(f"Ranking saved to: {output_path}")
        print("="*60)
        print(df_result.head(20))