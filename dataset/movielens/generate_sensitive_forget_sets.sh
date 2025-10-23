#!/bin/bash

# Configuration
dataset="movielens.inter"
rating_threshold=3.5

# Define sensitive movie categories and their genre/tag keywords
# You can customize these categories and keywords
declare -a categories=("horror" "romance")
declare -A categories_to_genres=(
    ["horror"]="Horror Thriller"
    ["romance"]="Romance"
)

# Random seeds for reproducibility
seeds=(2 3 5 7 11)

# Forget ratios (fraction of dataset to unlearn)
forget_ratios=(0.0001 0.00001 0.000001)

# Step 1: Identify sensitive movies for each category
echo "Step 1: Identifying sensitive movies by genre..."
for category in "${categories[@]}"; do
    echo "  Processing category: ${category}"
    python identify_sensitive_movies.py \
        --genres ${categories_to_genres[$category]} \
        --output "sensitive_movies_${category}.txt"
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
    sensitive_file="sensitive_movies_${category}.txt"

    # Check if sensitive movies file exists and has content
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
echo "Done! Generated forget sets for MovieLens dataset"
echo "Output files follow the pattern: ${dataset%.inter}_unlearn_pairs_sensitive_category_*_seed_*_unlearning_fraction_*.inter"
