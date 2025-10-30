#!/bin/bash

# Configuration
dataset="goodreads.inter"
rating_threshold=3.0

# Define sensitive book categories and their keywords
# You can customize these categories and keywords
declare -a categories=("mental_health")
declare -A categories_to_keywords=(
    ["mental_health"]="suicide suicidal depression depressed anxiety anxious ptsd trauma traumatic abuse abused violence violent rape sexual assault molest addiction alcoholism drug psychiatric psychological therapy counseling bipolar schizophrenia psychosis grief bereavement harm cutting disorder anorexia bulimia"
)

# Alternative: Use genre-based categories (commented out by default)
# Uncomment the lines below to use genre-based search instead
# declare -a categories=("romance" "horror")
# declare -A categories_to_genres=(
#     ["romance"]="romance erotica"
#     ["horror"]="horror thriller"
# )
# use_genres=true

# Random seeds for reproducibility
seeds=(2 3 5 7 11)

# Forget ratios (fraction of dataset to unlearn)
forget_ratios=(0.0001 0.00001 0.000001)

# Determine search mode
use_genres=${use_genres:-false}

# Step 1: Identify sensitive books for each category
echo "Step 1: Identifying sensitive books..."
for category in "${categories[@]}"; do
    echo "  Processing category: ${category}"

    if [ "$use_genres" = true ]; then
        # Genre-based search
        python identify_sensitive_books.py \
            --use-genres \
            --genres ${categories_to_genres[$category]} \
            --output "sensitive_books_${category}.txt"
    else
        # Keyword-based search (default)
        python identify_sensitive_books.py \
            --keywords ${categories_to_keywords[$category]} \
            --output "sensitive_books_${category}.txt"
    fi
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
    sensitive_file="sensitive_books_${category}.txt"

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
