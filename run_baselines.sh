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

        echo "Submitting job with model: $model, dataset: $dataset, seed: $seed, config: $config_file"
        python run_recbole.py --model $model --dataset $dataset --seed $seed --config_files $config_file
    done
done