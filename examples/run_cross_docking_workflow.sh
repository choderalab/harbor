#!/bin/zsh
python ../harbor/scripts/run_full_cross_docking_evaluation.py \
--input /Users/alexpayne/Scientific_Projects/mers-drug-discovery/sars2-retrospective-analysis/20240424_multi_pose_docking_cross_docking/results_csvs/20240503_combined_results_with_data.csv \
--output analyzed_data/ \
--n_cpus 4 \
--parameters ../harbor/scripts/settings.yml