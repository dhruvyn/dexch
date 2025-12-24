import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import argparse
import os
import sys

# ==========================================
# UTILITIES: LOGGING & FILE MANAGEMENT
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

def setup_directories(parent_dir, threshold):
    """Creates a subdirectory for the specific threshold run."""
    run_dir = os.path.join(parent_dir, f"run_threshold_{threshold}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir

# ==========================================
# VISUAL ANALYSIS: CORRELATION HEATMAP
# ==========================================
def corr_heatmap(X, save_dir=None): 

    # 1. Select Numeric Columns
    numeric_df = X

    # 2. Compute Correlation (Spearman)
    print("Computing Spearman Correlation Matrix...")
    corr_matrix = numeric_df.corr(method='spearman')

    # 3. Plotting
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='coolwarm', 
        center=0, 
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        vmin=-1, vmax=1 
    )

    plt.title('Feature Correlation Matrix (Spearman)', fontsize=20)
    plt.yticks(rotation=0)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
        plt.close() # Close to free memory
    else:
        plt.show()

    # 4. Numeric Output
    print("\n--- HIGHLY CORRELATED PAIRS (>0.95) ---")
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1.0] 

    return corr_pairs

# ==========================================
# HIERARCHICAL CLUSTERING ANALYSIS
# ==========================================
def clustering_analysis(X, threshold=0.5, plot=True, save_dir=None): 
    # 2. Calculate Correlation Distance
    corr = X.corr(method='spearman')
    distance_matrix = 1 - np.abs(corr)

    # 3. Perform Clustering (Linkage)
    linkage_matrix = sch.linkage(squareform(distance_matrix), method='ward')

    # 4. Plot Dendrogram
    if plot: 
        plt.figure(figsize=(15, 8))
        dendrogram = sch.dendrogram(
            linkage_matrix,
            labels=X.columns,
            leaf_rotation=90,
            leaf_font_size=10
        )
        plt.title(f'Feature Clustering Dendrogram (Ward Linkage, T={threshold})', fontsize=20)
        plt.xlabel('Features')
        plt.ylabel('Distance (1 - |Corr|)')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "dendrogram.png"))
            plt.close()
        else:
            plt.show()

    return linkage_matrix

# ==========================================
# FINAL CLUSTER ASSIGNMENT 
# ==========================================
def make_clusters(X, linkage_matrix, threshold=0.5, verbose=True):

    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # 2. Create the Final Map
    feat_cluster_map = pd.DataFrame({
        'Feature': X.columns,
        'Cluster_ID': cluster_labels
    })

    # 3. Save/Print for Reference
    if verbose:
        print(f"--- FINAL CLUSTER CONFIGURATION (T={threshold}) ---")
        print(f"Total Clusters: {feat_cluster_map['Cluster_ID'].nunique()}")

    feat_cluster_map = feat_cluster_map.sort_values('Cluster_ID')
    feature_to_cluster = dict(zip(feat_cluster_map['Feature'], feat_cluster_map['Cluster_ID']))

    if verbose:
        for cid in sorted(feat_cluster_map['Cluster_ID'].unique()):
            feats = feat_cluster_map[feat_cluster_map['Cluster_ID'] == cid]['Feature'].tolist()
            print(f"\nCluster {cid} ({len(feats)} features):")
            print(feats)
            
    return feature_to_cluster, feat_cluster_map

def align_data(df_final, targets, common_keys=['BAR_TIMESTAMP', 'SYMBOL', 'EXCHANGE'], verbose=True):
    # ==============================================================================
    # STEP 1: ROBUST ALIGNMENT & CLEANUP
    # ==============================================================================
    if verbose:
        print("--- ALIGNING FEATURES AND TARGETS ---")

    df_aligned = pd.merge(
        df_final, 
        targets, 
        on=common_keys, 
        how='inner'
    )

    target_cols = [c for c in targets.columns if c not in common_keys]
    feature_cols = [c for c in df_final.columns if c not in common_keys and c not in target_cols]

    X_full = df_aligned[feature_cols].copy()
    y_full = df_aligned[target_cols].copy()
    
    # Random Forest cannot handle strings.
    X_numeric = X_full.select_dtypes(include=[np.number])

    if verbose: 
        print(f"Aligned Rows: {len(X_numeric)}")
        print(f"Feature Count: {X_numeric.shape[1]} (Numeric Only)")
        print(f"Targets: {y_full.columns.tolist()}")
    
    return X_numeric, y_full

def verify_data(X_numeric, y_full, feature_to_cluster, verbose=True):
    feature_to_cluster_clean = {k: v for k, v in feature_to_cluster.items() if k in X_numeric.columns}
    return feature_to_cluster_clean

# ==============================================================================
# ENGINE: ROBUST CLUSTERED MDA (WALK-FORWARD)
# ==============================================================================
def get_robust_clustered_mda(X, y, feature_to_cluster, n_splits=3, purge_size=24):
    
    # 1. Setup Model (Light RF)
    model_params = {'n_estimators': 150, 'max_depth': 5, 'n_jobs': -1, 'random_state': 42}
    
    # 2. Walk-Forward Splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    print(f"Starting {n_splits}-Fold Walk-Forward MDA (Purge={purge_size})...")
    
    for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X)):
        
        # --- PURGING ---
        if len(val_indices) <= purge_size:
            continue
        val_indices = val_indices[purge_size:]
        
        # Split
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
        
        # Fit
        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)
        
        # Base Score (Log Loss)
        base_probs = clf.predict_proba(X_val)
        base_score = log_loss(y_val, base_probs, labels=clf.classes_)
        
        # --- CLUSTER PERMUTATION ---
        fold_importances = {}
        unique_clusters = sorted(list(set(feature_to_cluster.values())))
        
        for cid in unique_clusters:
            cluster_feats = [f for f, c in feature_to_cluster.items() if c == cid and f in X.columns]
            if not cluster_feats: continue
            
            # Shuffle
            X_val_shuffled = X_val.copy()
            shuffled_vals = X_val[cluster_feats].values.copy()
            np.random.shuffle(shuffled_vals)
            X_val_shuffled[cluster_feats] = shuffled_vals
            
            # Score Drop
            shuffled_probs = clf.predict_proba(X_val_shuffled)
            shuffled_score = log_loss(y_val, shuffled_probs, labels=clf.classes_)
            
            fold_importances[cid] = shuffled_score - base_score
            
        fold_scores.append(fold_importances)
        print(f"  Fold {fold_idx+1} Base Loss: {base_score:.4f}")

    return pd.DataFrame(fold_scores).mean().to_dict()

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================
def run_pipeline(X_input, y_input, threshold=0.5, n_splits=3, purge_size=24, verbose=True, save_dir=None):
    
    X_num = X_input.select_dtypes(include=[np.number])
    
    # Plotting Logic (saves to save_dir)
    linkage_matrix = clustering_analysis(X_num, threshold=threshold, plot=True, save_dir=save_dir)

    # use linkage_matrix to make clusters
    feature_to_cluster, _ = make_clusters(X_num, linkage_matrix, threshold=threshold, verbose=verbose)

    # align data 
    X_numeric, y_full = align_data(X_input, y_input, verbose=verbose)
    
    # verify data
    feature_to_cluster_clean = verify_data(X_numeric, y_full, feature_to_cluster, verbose=verbose)

    mda_results = {}

    for target_col in y_full.columns:
        print(f"\n--- Processing Target: {target_col} ---")
        
        y_series = y_full[target_col]
        valid_mask = ~y_series.isna()
        
        X_clean = X_numeric[valid_mask]
        y_clean = y_series[valid_mask]
        
        scores = get_robust_clustered_mda(
            X_clean, 
            y_clean, 
            feature_to_cluster_clean, 
            n_splits=n_splits, 
            purge_size=purge_size
        )
        
        mda_results[target_col] = scores

    # ==============================================================================
    # FINAL REPORT
    # ==============================================================================
    df_mda = pd.DataFrame(mda_results)
    df_mda.index.name = "Cluster_ID"
    df_mda = df_mda.sort_index()
    
    if verbose:
        print("\n--- FINAL CLUSTER IMPORTANCE (Log Loss Decrease) ---")
        print(df_mda)

    # Check for "Dead" Clusters
    df_mda['Total_Importance'] = df_mda.sum(axis=1)
    dead_clusters = df_mda[df_mda['Total_Importance'] <= 0].index.tolist()

    return df_mda, dead_clusters

def visualize_cluster_importance(df_mda, dead_clusters, save_dir=None):
    # ==============================================================================
    # VISUALIZE MDA RESULTS
    # ==============================================================================

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- PLOT A: HEATMAP ---
    sns.heatmap(
        df_mda.drop(columns=['Total_Importance'], errors='ignore'), 
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0,
        ax=axes[0],
        cbar_kws={'label': 'Log Loss Decrease (Higher is Better)'}
    )
    axes[0].set_title('Cluster Importance by Target', fontsize=15)
    axes[0].set_ylabel('Cluster ID')
    axes[0].set_xlabel('Target')

    # --- PLOT B: RANKED BAR CHART ---
    total_importance = df_mda.drop(columns=['Total_Importance'], errors='ignore').sum(axis=1).sort_values(ascending=True)
    colors = ['red' if x < 0 else 'forestgreen' for x in total_importance.values]

    total_importance.plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
    axes[1].axvline(0, color='black', linewidth=1) 
    axes[1].set_title('Overall Cluster Importance (Sum Across Targets)', fontsize=15)
    axes[1].set_xlabel('Cumulative Log Loss Decrease')

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "mda_importance_results.png"))
        plt.close()
    else:
        plt.show()

    # ==============================================================================
    # TEXT SUMMARY
    # ==============================================================================
    print("\n--- DECISION REPORT ---")
    best_cluster = total_importance.idxmax()
    print(f"🏆 MVP Cluster: Cluster {best_cluster} (Highest Total Signal)")

    negative_clusters = total_importance[total_importance < 0].index.tolist()
    if negative_clusters:
        print(f"⚠️  Harmful Clusters (Negative Score - REMOVE THESE): {negative_clusters}")
    else:
        print("✅ No clusters were explicitly harmful (all > 0).")

    useless_clusters = total_importance[(total_importance >= 0) & (total_importance < 1e-4)].index.tolist()
    if useless_clusters:
        print(f"💤 Weak/Noise Clusters (Near Zero Impact): {useless_clusters}")

    if dead_clusters:
        print(f"💀 Dead Clusters (Negative or Zero across all targets): {dead_clusters}")
    else:
        print("✅ No clusters were dead (all > 0 across all targets).")

DEFAULT_TARGET_FILE_PATH = "./sample_data/btc_5min_targets.csv"
DEFAULT_THRESHOLD_RUNS = [0.5, 0.7]

def setup_directories(parent_dir, threshold):
    """Creates a subdirectory for the specific threshold run."""
    run_dir = os.path.join(parent_dir, f"run_threshold_{threshold}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run analysis on a dataframe split")

    parser.add_argument("--X_path", type=str, required=True, help="Path to X CSV file")
    parser.add_argument("--split_dir", type=str, required=True, help="Base path for analysis dump")
    parser.add_argument("--target_path", default=DEFAULT_TARGET_FILE_PATH, type=str,required=True, help="Path to target CSV file")
    parser.add_argument(
        "--thresholds", 
        nargs='+',           # Reads one or more arguments into a list
        type=float,          # Converts each argument to a float
        default= DEFAULT_THRESHOLD_RUNS,  # Default value 
        help="List of thresholds (usage: --thresholds 0.3 0.5 0.7)"
    )
    args = parser.parse_args()
    
    # Define thresholds to iterate over
    thresholds = args.thresholds
    TARGET_FILE_PATH = args.target_path

    # ---------------------------------------------
    # 1. SETUP BASE DIRECTORY
    # ---------------------------------------------
    # Use the provided dump_path as the root for analysis_runs
    base_output_dir = os.path.join(args.split_dir, "analysis_runs")
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Load Data ONCE
    print(f"Loading X from: {args.X_path}")
    df_final = pd.read_csv(args.X_path)
        
    X = df_final.select_dtypes(include=[np.number])
    targets = pd.read_csv(TARGET_FILE_PATH)

    # ---------------------------------------------
    # RUN LOOP FOR EACH THRESHOLD
    # ---------------------------------------------
    for thresh in thresholds:
        # Setup subdirectory: split_dir/analysis_runs/run_threshold_X
        run_dir = setup_directories(base_output_dir, thresh)
        
        # Start Logging
        log_file = os.path.join(run_dir, "log.txt")
        sys.stdout = Logger(log_file)
        
        print(f"\n{'='*40}")
        print(f"STARTING RUN FOR THRESHOLD: {thresh}")
        print(f"Saving outputs to: {os.path.abspath(run_dir)}") # Print specific run path
        print(f"{'='*40}\n")
        
        try:
            # 3. Initial Heatmap (Saved to run_dir)
            # This is static per data, but we save it in every folder for completeness
            corr_heatmap(X, save_dir=run_dir)
            
            # 4. Run Pipeline (Clustering + MDA)
            df_mda, dead_clusters = run_pipeline(
                df_final, 
                targets, 
                threshold=thresh, 
                verbose=True, 
                save_dir=run_dir # Pass dir to save dendrogram
            )
            
            # 5. Visualize MDA (Saved to run_dir)
            visualize_cluster_importance(df_mda, dead_clusters, save_dir=run_dir)
            
        except Exception as e:
            print(f"ERROR in run {thresh}: {e}")
        
        print(f"\nRun for threshold {thresh} completed. Results saved to {run_dir}")
        
        # 6. Stop Logging (Restore stdout)
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    print("\nAll threshold runs completed successfully.")