#!/bin/bash

SEEDS=(2 3 5 7 11)
fractions=(0.0001 0.001)
fraction_string="${fractions[*]}"
methods=(unpopular popular random)

for method in "${methods[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running unlearning sample selection with fractions=(${fraction_string}) and method=$method"
        time python pick_unlearning_samples.py --seed $SEED --unlearning_fractions $fraction_string --method $method
        echo "---------------------------------------------"
    done
done
