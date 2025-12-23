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

# ==========================================
# VISUAL ANALYSIS: CORRELATION HEATMAP
# ==========================================
def corr_heatmap(X) : 

    # 1. Select Numeric Columns
    numeric_df=  X

    # 2. Compute Correlation (Spearman)
    # Spearman is robust to outliers and captures monotonic relationships better than Pearson
    print("Computing Spearman Correlation Matrix...")
    corr_matrix = numeric_df.corr(method='spearman')

    # 3. Plotting
    plt.figure(figsize=(20, 16))

    # We use a mask to hide the upper triangle (it's symmetrical, so redundant)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Clustermap vs Heatmap
    # A standard heatmap is good, but a 'clustermap' automatically reorders 
    # columns to put similar features next to each other.
    # We'll use a standard heatmap first for raw inspection.
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='coolwarm',     # Red = High Positive, Blue = High Negative
        center=0,            # Center colormap at 0 correlation
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        vmin=-1, vmax=1      # Fix scale to -1 to 1
    )

    plt.title('Feature Correlation Matrix (Spearman)', fontsize=20)
    plt.yticks(rotation=0)
    plt.show()

    # 4. Numeric Output: Identify Top Collinear Pairs
    # This prints the pairs with correlation > 0.95 (Candidates for dropping)
    print("\n--- HIGHLY CORRELATED PAIRS (>0.95) ---")
    # Unstack the matrix to get pairs
    corr_pairs = corr_matrix.abs().unstack()
    # Filter self-correlations and duplicates
    corr_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1.0] # Remove self correlation

    # Show top 20
    return corr_pairs
# ==========================================
# HIERARCHICAL CLUSTERING ANALYSIS
# ==========================================
def clustering_analysis(X, threshold=0.5, plot=True) : 
    # 2. Calculate Correlation Distance
    # We use Spearman (Rank) correlation
    corr = X.corr(method='spearman')

    # Convert correlation to "distance"
    # Distance is low (close) if correlation is high.
    # We use absolute correlation because -0.9 is just as strong a relationship as 0.9.
    distance_matrix = 1 - np.abs(corr)

    # 3. Perform Clustering (Linkage)
    # Ward's method minimizes the variance within clusters being merged.
    # It tends to create compact, even-sized clusters.
    # We use squareform to convert the matrix to the format scipy expects.
    
    linkage_matrix = sch.linkage(squareform(distance_matrix), method='ward')

    
    # 4. Plot Dendrogram
    if plot : 
        plt.figure(figsize=(15, 8))
        dendrogram = sch.dendrogram(
            linkage_matrix,
            labels=X.columns,
            leaf_rotation=90,
            leaf_font_size=10
        )
        plt.title('Feature Clustering Dendrogram (Ward Linkage)', fontsize=20)
        plt.xlabel('Features')
        plt.ylabel('Distance (1 - |Corr|)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold = 0.5 (Corr > 0.5)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return linkage_matrix

# ==========================================
# FINAL CLUSTER ASSIGNMENT 
# ==========================================
def make_clusters(X, linkage_matrix, threshold=0.5, verbose = True):

    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # 2. Create the Final Map
    feat_cluster_map = pd.DataFrame({
        'Feature': X.columns,
        'Cluster_ID': cluster_labels
    })

    # 3. Save/Print for Reference
    if verbose :
        print(f"--- FINAL CLUSTER CONFIGURATION (T={threshold}) ---")
        print(f"Total Clusters: {feat_cluster_map['Cluster_ID'].nunique()}")

    # Sort by cluster for easy reading
    feat_cluster_map = feat_cluster_map.sort_values('Cluster_ID')

    # Create a dictionary for quick lookups later {Feature: Cluster_ID}
    feature_to_cluster = dict(zip(feat_cluster_map['Feature'], feat_cluster_map['Cluster_ID']))

    # Show the breakdown
    if verbose:
        for cid in sorted(feat_cluster_map['Cluster_ID'].unique()):
            feats = feat_cluster_map[feat_cluster_map['Cluster_ID'] == cid]['Feature'].tolist()
            print(f"\nCluster {cid} ({len(feats)} features):")
            print(feats)
    return feature_to_cluster, feat_cluster_map

def align_data(df_final, targets, common_keys = ['BAR_TIMESTAMP', 'SYMBOL', 'EXCHANGE'], verbose = True):
    # ==============================================================================
    # STEP 1: ROBUST ALIGNMENT & CLEANUP
    # ==============================================================================
    if verbose :
        print("--- ALIGNING FEATURES AND TARGETS ---")

    # 1. Merge Features (df_final) and Targets on Metadata keys
    # This ensures we only keep rows where we have BOTH features AND labels.

    # Perform Inner Join
    df_aligned = pd.merge(
        df_final, 
        targets, 
        on=common_keys, 
        how='inner'
    )

    # 2. Separate X (Features) and Y (Targets)
    # Identify Target Columns (assuming they are the ones from the 'targets' df)
    target_cols = [c for c in targets.columns if c not in common_keys]
    feature_cols = [c for c in df_final.columns if c not in common_keys and c not in target_cols]

    X_full = df_aligned[feature_cols].copy()
    y_full = df_aligned[target_cols].copy()
    
    # 3. DROP NON-NUMERIC COLUMNS from X
    # Random Forest cannot handle 'SYMBOL' or 'EXCHANGE' strings.
    # We select only numeric types (float/int).
    X_numeric = X_full.select_dtypes(include=[np.number])

    # Sanity Check
    if verbose : 
        print(f"Aligned Rows: {len(X_numeric)}")
        print(f"Feature Count: {X_numeric.shape[1]} (Numeric Only)")
        print(f"Targets: {y_full.columns.tolist()}")
    
    return X_numeric, y_full

def verify_data(X_numeric, y_full, feature_to_cluster, verbose = True):
    # Ensure our 'feature_to_cluster' map doesn't contain columns we just dropped (strings)
    feature_to_cluster_clean = {k: v for k, v in feature_to_cluster.items() if k in X_numeric.columns}
    return feature_to_cluster_clean


# ==============================================================================
# ENGINE: ROBUST CLUSTERED MDA (WALK-FORWARD)
# ==============================================================================
def get_robust_clustered_mda(X, y, feature_to_cluster, n_splits=3, purge_size=24):
    """
    Args:
        purge_size: 24 (Matches your Triple Barrier Expiry)
    """
    # 1. Setup Model (Light RF)
    model_params = {'n_estimators': 150, 'max_depth': 5, 'n_jobs': -1, 'random_state': 42}
    
    # 2. Walk-Forward Splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    print(f"Starting {n_splits}-Fold Walk-Forward MDA (Purge={purge_size})...")
    
    for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X)):
        
        # --- PURGING ---
        # Skip the first 'purge_size' rows of validation to prevent overlap leakage
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
            # Get features for this cluster that exist in X
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

    # Aggregate
    return pd.DataFrame(fold_scores).mean().to_dict()

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================
def run_pipeline(X_input, y_input, threshold = 0.5, n_splits=3, purge_size=24, verbose = True):
    
    X_num= X_input.select_dtypes(include=[np.number])
    linkage_matrix= clustering_analysis(X_num, plot = False)

    # use linkage_matrix to make clusters (make_clusters())
    feature_to_cluster, _ = make_clusters(X_num, linkage_matrix, threshold = threshold, verbose = verbose)

    # align data 
    X_numeric, y_full = align_data(X_input, y_input, verbose = verbose)
    
    # verify data
    feature_to_cluster_clean = verify_data(X_numeric, y_full, feature_to_cluster, verbose = verbose)

    mda_results = {}

    for target_col in y_full.columns:
        print(f"\n--- Processing Target: {target_col} ---")
        
        # Prepare Y (Drop NaNs)
        y_series = y_full[target_col]
        valid_mask = ~y_series.isna()
        
        # Filter X and Y to match valid targets
        X_clean = X_numeric[valid_mask]
        y_clean = y_series[valid_mask]
        
        # Run MDA
        # Note: purge_size=24 matches your expiry parameters
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
    if verbose :
        print("\n--- FINAL CLUSTER IMPORTANCE (Log Loss Decrease) ---")
        print(df_mda)

    # Check for "Dead" Clusters (Negative or Zero score across all targets)
    df_mda['Total_Importance'] = df_mda.sum(axis=1)
    dead_clusters = df_mda[df_mda['Total_Importance'] <= 0].index.tolist()


    return df_mda, dead_clusters

def visualize_cluster_importance(df_mda, dead_clusters):
    # ==============================================================================
    # VISUALIZE MDA RESULTS
    # ==============================================================================

    # 1. Setup the figure layout (2 plots: Heatmap & Bar Chart)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- PLOT A: HEATMAP (Target Specificity) ---
    # This shows exactly which cluster drives which target signal.
    sns.heatmap(
        df_mda.drop(columns=['Total_Importance'], errors='ignore'), # Exclude total if it exists
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",   # Red = Bad (Negative), Green = Good (Positive)
        center=0,        # Center the color scale at 0 to highlight negative scores
        ax=axes[0],
        cbar_kws={'label': 'Log Loss Decrease (Higher is Better)'}
    )
    axes[0].set_title('Cluster Importance by Target', fontsize=15)
    axes[0].set_ylabel('Cluster ID')
    axes[0].set_xlabel('Target')

    # --- PLOT B: RANKED BAR CHART (Overall Power) ---
    # Calculate Total Importance across all targets to find the "MVPs"
    total_importance = df_mda.drop(columns=['Total_Importance'], errors='ignore').sum(axis=1).sort_values(ascending=True)

    # Color bars based on Positive (Green) or Negative (Red) impact
    colors = ['red' if x < 0 else 'forestgreen' for x in total_importance.values]

    total_importance.plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
    axes[1].axvline(0, color='black', linewidth=1) # Zero line
    axes[1].set_title('Overall Cluster Importance (Sum Across Targets)', fontsize=15)
    axes[1].set_xlabel('Cumulative Log Loss Decrease')

    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # TEXT SUMMARY
    # ==============================================================================
    print("\n--- DECISION REPORT ---")
    # 1. Best Cluster
    best_cluster = total_importance.idxmax()
    print(f"🏆 MVP Cluster: Cluster {best_cluster} (Highest Total Signal)")

    # 2. Worst Clusters (Negative Impact)
    negative_clusters = total_importance[total_importance < 0].index.tolist()
    if negative_clusters:
        print(f"⚠️  Harmful Clusters (Negative Score - REMOVE THESE): {negative_clusters}")
    else:
        print("✅ No clusters were explicitly harmful (all > 0).")

    # 3. Useless Clusters (Near Zero)
    # Clusters that add less than 0.0001 total log loss improvement are basically noise.
    useless_clusters = total_importance[(total_importance >= 0) & (total_importance < 1e-4)].index.tolist()
    if useless_clusters:
        print(f"💤 Weak/Noise Clusters (Near Zero Impact): {useless_clusters}")

    # 4. Dead Clusters (Negative or Zero across all targets)
    if dead_clusters:
        print(f"💀 Dead Clusters (Negative or Zero across all targets): {dead_clusters}")
    else:
        print("✅ No clusters were dead (all > 0 across all targets).")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run analysis on a dataframe split"
    )

    parser.add_argument(
        "--X_path",
        type=str,
        required=True,
        help="Path to X CSV file"
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        required=True,
        help="Start index of the split"
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        required=True,
        help="End index of the split"
    )

    args = parser.parse_args()

    # ---- use args ----
    print(f"Loading X from: {args.X_path}")
    print(f"Split indices: {args.start_idx} → {args.end_idx}")
    
    
    df_final = pd.read_csv('df_final.csv')
    X = df_final.select_dtypes(include=[np.number])
    print(df_final.head())

    targets= pd.read_csv("./sample_data/btc_5min_targets.csv")



    corr= corr_heatmap(X)
    linkage_matrix= clustering_analysis(X)


    df_mda , dead_clusters = run_pipeline(df_final, targets, threshold = 0.5, verbose = True)
    visualize_cluster_importance(df_mda, dead_clusters)