#!/bin/bash

sensitive_items=("sensitive_asins_meat.txt" "sensitive_asins_alcohol.txt")
seeds=(2 3 5 7 11)
forget_ratios=(0.0001 0.00001 0.000001)
categories=("meat" "alcohol")

declare -A categories_to_keywords=( ["meat"]="meat beef pork chicken lamb turkey bacon ham sausage salami ham" ["alcohol"]="alcohol beer wine liquor whisky whiskey vodka gin rum bourbon tequila brandy cognac IPA pilsner champagne prosecco" )

for category in "${categories[@]}"; do
    python identify_sensitive_items.py --output "sensitive_parent_asins_${category}.txt" --categories "${categories_to_keywords[$category]}"
    python get_sensitive_asins_from_sensitive_parent_asins.py --input "sensitive_parent_asins_${category}.txt" --output "sensitive_asins_${category}.txt"
done


for sensitive_item in "${sensitive_items[@]}"; do
    for seed in "${seeds[@]}"; do
        for forget_ratio in "${forget_ratios[@]}"; do
            echo "Generating forget sets for sensitive items: ${sensitive_item}, seed: ${seed}, forget ratio: ${forget_ratio}"
            python generate_forget_sets.py --sensitive-items $sensitive_item --seed $seed --forget-ratio $forget_ratio
        done
    done
done
