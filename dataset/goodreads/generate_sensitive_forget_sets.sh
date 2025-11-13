#!/bin/bash

# Configuration
dataset="goodreads.inter"
rating_threshold=3.0

# Define sensitive book categories
# Available categories: mental_health, violence, explicit
# The new multi-signal detection uses titles, descriptions, shelves, and genres for better accuracy
declare -a categories=("mental_health")

# Random seeds for reproducibility
seeds=(2 3 5 7 11)

# Forget ratios (fraction of dataset to unlearn)
forget_ratios=(0.0001 0.00001 0.000001)

# Step 1: Identify sensitive books for each category using multi-signal detection
echo "Step 1: Identifying sensitive books using multi-signal detection..."
echo "  (Searching in titles, descriptions, shelves, and genres)"
for category in "${categories[@]}"; do
    echo "  Processing category: ${category}"

    sensitive_file="sensitive_asins_${category}.txt"

    # Skip if file already exists
    if [ -f "$sensitive_file" ]; then
        echo "    ${sensitive_file} already exists, skipping..."
        continue
    fi

    # Use the new category-based approach with multi-signal detection
    # This requires at least 2 signals (e.g., title + description, or title + shelf)
    # to reduce false positives from popular books
    python identify_sensitive_books.py \
        --category "${category}" \
        --output "$sensitive_file"
    
    # For even stricter matching, you can increase min_signals:
    # python identify_sensitive_books.py \
    #     --category "${category}" \
    #     --min-signals 3 \
    #     --output "$sensitive_file"
done

# Step 2: Copy generate_forget_sets.py if not present
if [ ! -f "generate_forget_sets.py" ]; then
    echo "Copying generate_forget_sets.py from amazon_reviews_grocery_and_gourmet_food..."
    cp ../amazon_reviews_grocery_and_gourmet_food/generate_forget_sets.py .
fi

# Step 3: Generate forget sets for each combination of category, seed, and forget ratio
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
echo "Done! Generated forget sets for Goodreads dataset"
echo "Output files follow the pattern: ${dataset%.inter}_unlearn_pairs_sensitive_category_*_seed_*_unlearning_fraction_*.inter"
