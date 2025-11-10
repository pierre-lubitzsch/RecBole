#!/bin/bash

# Configuration
dataset="amazon_reviews_books.inter"
rating_threshold=3.0

# Define sensitive book categories and their keywords
declare -a categories=("health" "violence")
declare -A categories_to_keywords=(
    ["health"]="suicide depression anxiety \"mental health\" \"mental illness\" \"self harm\" trauma ptsd"
    ["violence"]="violence murder war torture abuse assault weapon blood"
)

# Random seeds for reproducibility
seeds=(2 3 5 7 11)

# Forget ratios (fraction of dataset to unlearn)
forget_ratios=(0.0001 0.00001 0.000001)

# Metadata file for identifying sensitive items
metadata_file="meta_Books.jsonl.gz"

# Step 1: Identify sensitive books for each category
echo "Step 1: Identifying sensitive books..."
for category in "${categories[@]}"; do
    echo "  Processing category: ${category}"

    sensitive_file="sensitive_asins_${category}.txt"

    # Skip if file already exists
    if [ -f "$sensitive_file" ]; then
        echo "    ${sensitive_file} already exists, skipping..."
        continue
    fi

    # Keyword-based search
    python identify_sensitive_items.py \
        --keywords ${categories_to_keywords[$category]} \
        --output "$sensitive_file" \
        --files "$metadata_file"
done

# Step 2: Generate forget sets for each combination of category, seed, and forget ratio
echo ""
echo "Step 2: Generating forget sets..."
for category in "${categories[@]}"; do
    sensitive_file="sensitive_asins_${category}.txt"

    # Check if sensitive books file exists and has content
    if [ ! -f "$sensitive_file" ]; then
        echo "Warning: ${sensitive_file} not found, skipping category ${category}"
        continue
    fi

    if [ ! -s "$sensitive_file" ]; then
        echo "Warning: ${sensitive_file} is empty, skipping category ${category}"
        continue
    fi

    for seed in "${seeds[@]}"; do
        for forget_ratio in "${forget_ratios[@]}"; do
            echo "  Generating: category=${category}, seed=${seed}, ratio=${forget_ratio}"
            python generate_forget_sets.py \
                --sensitive-items "$sensitive_file" \
                --seed $seed \
                --forget-ratio $forget_ratio \
                --dataset $dataset \
                --rating-threshold $rating_threshold
        done
    done
done

echo ""
echo "Done! Generated forget sets for Amazon Reviews Books dataset"
echo "Output files follow the pattern: ${dataset%.inter}_unlearn_pairs_sensitive_category_*_seed_*_unlearning_fraction_*.inter"
