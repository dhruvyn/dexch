

* Run Scripts Example  on  XRPUSDT_data: 

    - generate nan_cleaned_data and data quality analysis

        python .\scripts\nan_clean_data_refactor.py --output_dir "./data/XRPUSDT_results" --output_filename "./data/XRPUSDT_5min_final_cleaned.csv" --plot_dir "./data/XRPUSDT_results/pipeline_artifacts" --liq_path "./data/XRPUSDT_data/5min_liq.csv" --deriv_path "./data/XRPUSDT_data/5min_deriv.csv" --trades_path "./data/XRPUSDT_data/5min_trades.csv"

    - generate targets

        python .\scripts\make_targets.py --input "./data/XRPUSDT_results/XRPUSDT_5min_final_cleaned.csv"  --out_dir ".\data\XRPUSDT_results" --out_name "XRPUSDT_5min_targets.csv" --expiry 24 --vol_lookback 100 --tb_width 1.5 --dyn_mult 1.0

    - generate feature importance analysis

        python .\scripts\split_run_analysis.py --filepath "./data/XRPUSDT_results/XRPUSDT_5min_final_cleaned.csv"  --output_folder "./data/XRPUSDT_results/recent" --latest_num_months 8 --target_path ".\data\XRPUSDT_results\XRPUSDT_5min_targets.csv"

    - aggregate results 

        python .\scripts\agg_new.py ".\data\BTCUSDT_results\btc_recent\split_recent_8m\analysis_runs\run_threshold_0.5\log.txt" ".\data\ETHUSDT_results\recent\split_recent_8m\analysis_runs\run_threshold_0.5\log.txt" ".\data\XRPUSDT_results\recent\split_recent_8m\analysis_runs\run_threshold_0.5\log.txt" --names "btc" "eth" "xrp"

 
C:\Users\Dhruv\Desktop\delta1\data\BTCUSDT_results\recent_12exp_1stdev_1.5stdev