

* Run Scripts Example  on  XRPUSDT_data: 

    - generate nan_cleaned_data and data quality analysis

        python .\scripts\nan_clean_data_refactor.py --output_dir "./XRPUSDT_results" --output_filename "./XRPUSDT_5min_final_cleaned.csv" --plot_dir "./XRPUSDT_results/pipeline_artifacts" --liq_path "./XRPUSDT_data/5min_liq.csv" --deriv_path "./XRPUSDT_data/5min_deriv.csv" --trades_path "./XRPUSDT_data/5min_trades.csv"

    - generate targets

        python .\scripts\make_targets.py --input "./XRPUSDT_results/XRPUSDT_5min_final_cleaned.csv"  --out_dir ".\XRPUSDT_results" --out_name "XRPUSDT_5min_targets.csv" --expiry 24 --vol_lookback 100 --tb_width 1.5 --dyn_mult 1.0

    - generate feature importance analysis

        python .\scripts\split_run_analysis.py --filepath "./XRPUSDT_results/XRPUSDT_5min_final_cleaned.csv"  --output_folder "./XRPUSDT_results/recent" --latest_num_months 8 --target_path ".\XRPUSDT_results\XRPUSDT_5min_targets.csv"

 
