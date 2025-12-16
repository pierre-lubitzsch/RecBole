#!/usr/bin/env python
"""
Standalone script to compute shadow models for RULI Privacy evaluation.

Usage:
    python compute_shadow_models_script.py --model BPR --dataset amazon_reviews --seed 42 --k 8
"""

import argparse
import sys
import os

# Add parent directory to path to import recbole modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recbole.quick_start.compute_shadow_models import compute_shadow_models


def main():
    parser = argparse.ArgumentParser(
        description="Compute shadow models for RULI Privacy evaluation"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of shadow models (must be even, default: 8)"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default=None,
        help="Config file paths (space-separated)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory to save shadow models (default: ./models)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0, but CUDA_VISIBLE_DEVICES takes precedence)"
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        help="Sensitive category for forget set (required if training on retain set)"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=None,
        help="Unlearning fraction for forget set (required if training on retain set)"
    )
    parser.add_argument(
        "--unlearning_sample_selection_seed",
        type=int,
        default=None,
        help="Seed for forget set selection (uses --seed if not specified)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="CF",
        choices=["CF", "SBR", "NBR"],
        help="Task type: CF (Collaborative Filtering), SBR (Session-Based), or NBR (Next Basket) (default: CF)"
    )
    
    args = parser.parse_args()
    
    # Parse config files
    config_file_list = None
    if args.config_files:
        config_file_list = args.config_files.strip().split(" ")
    
    # Prepare config dict
    config_dict = {
        "seed": args.seed,
        "model_dir": args.model_dir,
        "gpu_id": args.gpu_id,
        "task_type": args.task_type,
    }
    
    print(f"Computing {args.k} shadow models for {args.model} on {args.dataset}")
    print(f"Seed: {args.seed}, GPU ID: {args.gpu_id} (CUDA_VISIBLE_DEVICES takes precedence)")
    print(f"Shadow models will be saved to: ./saved/shadow_models/")
    
    try:
        saved_models, metadata = compute_shadow_models(
            model=args.model,
            dataset=args.dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            k=args.k,
            model_dir=args.model_dir,
            gpu_id=args.gpu_id,
            saved=True,
            sensitive_category=args.sensitive_category,
            unlearning_fraction=args.unlearning_fraction,
            unlearning_sample_selection_seed=args.unlearning_sample_selection_seed or args.seed,
            task_type=args.task_type,
        )
        
        print(f"\nSuccessfully computed {len(saved_models)} shadow models:")
        for model_path in saved_models:
            print(f"  - {model_path}")
        
        print(f"\nMetadata saved to: {metadata.get('metadata_path', 'N/A')}")
        
    except Exception as e:
        print(f"Error computing shadow models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

