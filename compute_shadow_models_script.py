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
    parser.add_argument(
        "--create_unlearned_models",
        action="store_true",
        help="Create unlearned versions of shadow models (Qh distribution) for proper RULI evaluation"
    )
    parser.add_argument(
        "--unlearning_algorithm",
        type=str,
        default=None,
        choices=["scif", "kookmin", "fanchuan", "gif", "ceu", "idea", "seif"],
        help="Unlearning algorithm to use when creating unlearned shadow models (default: scif). Can specify multiple comma-separated algorithms or use --all_algorithms"
    )
    parser.add_argument(
        "--all_algorithms",
        action="store_true",
        help="Create unlearned shadow models for all valid unlearning algorithms (scif, kookmin, fanchuan, gif, ceu, idea, seif)"
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=None,
        help="Max norm for SCIF unlearning algorithm"
    )
    parser.add_argument(
        "--kookmin_init_rate",
        type=float,
        default=0.01,
        help="Initial rate for Kookmin unlearning algorithm (default: 0.01)"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.01,
        help="Damping parameter for GIF/CEU/IDEA unlearning algorithms (default: 0.01)"
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
    
    # Determine which algorithms to use
    all_valid_algorithms = ["scif", "kookmin", "fanchuan", "gif", "ceu", "idea", "seif"]
    if args.create_unlearned_models:
        if args.all_algorithms:
            algorithms_to_use = all_valid_algorithms
            print(f"Will create unlearned shadow models for ALL algorithms: {', '.join(algorithms_to_use)}")
        elif args.unlearning_algorithm:
            # Support comma-separated list
            if ',' in args.unlearning_algorithm:
                algorithms_to_use = [alg.strip() for alg in args.unlearning_algorithm.split(',')]
            else:
                algorithms_to_use = [args.unlearning_algorithm]
            print(f"Will create unlearned shadow models for algorithms: {', '.join(algorithms_to_use)}")
        else:
            # Default to scif if no algorithm specified
            algorithms_to_use = ["scif"]
            print(f"Will create unlearned shadow models using default algorithm: scif")
    else:
        algorithms_to_use = []
    
    try:
        saved_models, metadata, all_saved_unlearned_models = compute_shadow_models(
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
            create_unlearned_models=args.create_unlearned_models,
            unlearning_algorithms=algorithms_to_use,
            max_norm=args.max_norm,
            kookmin_init_rate=args.kookmin_init_rate,
            damping=args.damping,
        )
        
        print(f"\nSuccessfully computed {len(saved_models)} shadow models:")
        for model_path in saved_models:
            print(f"  - {model_path}")
        
        if args.create_unlearned_models and all_saved_unlearned_models:
            total_unlearned = sum(len(models) for models in all_saved_unlearned_models.values())
            print(f"\nSuccessfully computed {total_unlearned} unlearned shadow models across {len(all_saved_unlearned_models)} algorithms:")
            for algorithm, model_paths in all_saved_unlearned_models.items():
                print(f"  {algorithm.upper()}: {len(model_paths)} models")
                for model_path in model_paths[:3]:  # Show first 3
                    print(f"    - {model_path}")
                if len(model_paths) > 3:
                    print(f"    ... and {len(model_paths) - 3} more")
        
        print(f"\nMetadata saved to: {metadata.get('metadata_path', 'N/A')}")
        
    except Exception as e:
        print(f"Error computing shadow models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

