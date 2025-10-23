#!/bin/bash

# Run baseline models for implicit collaborative filtering
# These models are deterministic/memory-based, so seed doesn't matter
# We use seed=2 for consistency with other experiments

models=("Random" "Pop" "ItemKNN" "EASE")
datasets=("amazon_reviews_grocery_and_gourmet_food")
seed=2

echo "Running baseline models with seed=${seed}"
echo "Models: ${models[@]}"
echo "Dataset: ${datasets[@]}"
echo ""

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
        config_file="configs/config_${model_lower}.yaml"
        session_name="${model_lower}"
        log_file="cluster_log/${model_lower}_${dataset}_seed${seed}.log"

        echo "Starting tmux session '${session_name}' for model: $model"

        tmux new -d -s "${session_name}" bash -c "conda activate env; python run_recbole.py --model $model --dataset $dataset --seed $seed --config_files $config_file 2>&1 | tee $log_file"    done
    done
done