#!/bin/bash

# Goodreads dataset preprocessing script
# Converts goodreads_interactions.csv to RecBole .inter format

interactions_file="goodreads_interactions.csv"
dataset_name="goodreads"
min_rating=3.0  # Optional: only keep ratings >= 3.0 (adjust as needed, or set to empty to keep all)

# Step 1: Convert CSV to raw .inter format
if [ -n "$min_rating" ]; then
    echo "Converting interactions with minimum rating filter: ${min_rating}"
    python preprocess_goodreads.py \
        --input-file "${interactions_file}" \
        --output-file "${dataset_name}_raw.inter" \
        --min-rating "${min_rating}"
else
    echo "Converting interactions without rating filter"
    python preprocess_goodreads.py \
        --input-file "${interactions_file}" \
        --output-file "${dataset_name}_raw.inter"
fi

# Step 2: Deduplicate and filter (using the same deduplicate.py from amazon_reviews)
# Copy deduplicate.py if not present
if [ ! -f "deduplicate.py" ]; then
    echo "Copying deduplicate.py from amazon_reviews_grocery_and_gourmet_food..."
    cp ../amazon_reviews_grocery_and_gourmet_food/deduplicate.py .
fi

python deduplicate.py --input "${dataset_name}_raw.inter" --output "${dataset_name}.inter"

echo "Done! Generated ${dataset_name}.inter"
