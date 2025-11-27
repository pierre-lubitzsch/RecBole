#!/usr/bin/env python3
"""
Script to evaluate all original models (*_best.pth) for a given dataset.
Loads the dataset once and evaluates all models sequentially to avoid redundant dataset loading.
The output matches the evaluation output from quick_start.py when using run_recbole.py.
"""

import os
import sys
import argparse
import glob
import re
import torch
import numpy as np
from pathlib import Path
from logging import getLogger

# Add RecBole imports
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_environment,
)


def parse_original_model_filename(filename):
    """Parse original model filename to extract parameters.

    Pattern: model_{model}_seed_{seed}_dataset_{dataset}_best.pth

    Returns:
        dict with keys: model, dataset, seed, or None if not an original model
    """
    basename = os.path.basename(filename)

    # Pattern for normal training: model_{model}_seed_{seed}_dataset_{dataset}_best.pth
    normal_pattern = r"model_(\w+)_seed_(\d+)_dataset_([\w_]+)_best\.pth"
    match = re.match(normal_pattern, basename)
    if match:
        return {
            "model": match.group(1),
            "seed": int(match.group(2)),
            "dataset": match.group(3),
            "model_file": filename
        }

    return None


def find_original_models_for_dataset(dataset, checkpoint_dir="saved", model_filter=None):
    """Find all original model files (*_best.pth) for a given dataset.

    Args:
        dataset: Dataset name to filter by
        checkpoint_dir: Directory containing model checkpoints
        model_filter: Optional list of model names to filter by

    Returns:
        List of parsed model info dictionaries, sorted by model name then seed
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []

    # Find all .pth files
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    # Filter by dataset and parse
    models = []
    for model_file in model_files:
        parsed = parse_original_model_filename(model_file)
        if parsed and parsed["dataset"] == dataset:
            # If model_filter is provided, check if model name is in the filter list
            if model_filter is not None:
                if parsed["model"] not in model_filter:
                    continue

            models.append(parsed)

    # Sort by model name then seed
    models.sort(key=lambda x: (x["model"], x["seed"]))

    return models


def load_dataset_once(dataset_name, model_name, config_file_list=None, config_dict=None, sensitive_category=None, task_type=None):
    """Load the dataset once for reuse across multiple model evaluations.

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model (needed for config)
        config_file_list: List of config files
        config_dict: Dictionary of config overrides
        sensitive_category: Sensitive category for evaluation
        task_type: Task type ('CF' for collaborative filtering, 'SBR' for session-based rec)

    Returns:
        Tuple of (config, dataset, test_data)
    """
    # Create config
    config_dict = config_dict or {}
    config_dict["dataset"] = dataset_name
    config_dict["model"] = model_name
    if sensitive_category:
        config_dict["sensitive_category"] = sensitive_category
    if task_type:
        config_dict["task_type"] = task_type

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    # Initialize logger
    init_logger(config)
    logger = getLogger()

    # Initialize seed
    init_seed(config["seed"], config["reproducibility"])

    # Create dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = create_dataset(config)
    logger.info(str(dataset))

    # Create dataloaders
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, dataset, train_data, valid_data, test_data, logger


def evaluate_model(model_info, config, dataset, train_data, valid_data, test_data, logger, cuda_device="0"):
    """Evaluate a single model.

    Args:
        model_info: Dictionary containing model metadata
        config: RecBole config object
        dataset: Dataset object
        train_data: Training dataloader
        valid_data: Validation dataloader
        test_data: Test dataloader
        logger: Logger object
        cuda_device: CUDA device to use

    Returns:
        Dictionary with evaluation results
    """
    model_file = model_info["model_file"]
    model_name = model_info["model"]
    seed = model_info["seed"]

    logger.info("=" * 80)
    logger.info(f"Evaluating: {os.path.basename(model_file)}")
    logger.info(f"Model: {model_name}, Seed: {seed}")
    logger.info("=" * 80)

    # Update config for this specific model/seed
    config["model"] = model_name
    config["seed"] = seed
    config["gpu_id"] = int(cuda_device)

    # Re-initialize seed for this model
    init_seed(seed, config["reproducibility"])

    # Create model and trainer
    model = get_model(config["model"])(config, dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Load model checkpoint
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return None

    logger.info(f"Loading model from: {model_file}")
    checkpoint = torch.load(model_file, map_location=config["device"])
    trainer.model.load_state_dict(checkpoint["state_dict"])
    if "other_parameter" in checkpoint:
        trainer.model.load_other_parameter(checkpoint["other_parameter"])
    trainer.model.eval()

    # Evaluate on test set
    logger.info("Running test evaluation...")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    # Log results
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "model": model_name,
        "seed": seed,
        "dataset": dataset.dataset_name,
        "test_result": test_result,
        "model_file": model_file,
    }

    # Sensitive item evaluation (if configured)
    if "sensitive_category" in config and config['sensitive_category'] is not None:
        logger.info("\nSensitive Item Evaluation for Original Model")

        sensitive_category = config['sensitive_category']
        sensitive_items_path = os.path.join(config['data_path'], f"sensitive_asins_{sensitive_category}.txt")

        if os.path.exists(sensitive_items_path):
            with open(sensitive_items_path, 'r') as f:
                sensitive_asins = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(sensitive_asins)} sensitive items from {sensitive_items_path}")

            # Map ASINs to internal item IDs
            sensitive_item_ids = set()
            iid_field = dataset.iid_field
            for asin in sensitive_asins:
                try:
                    item_id = dataset.token2id(iid_field, asin)
                    sensitive_item_ids.add(item_id)
                except ValueError:
                    # ASIN not in dataset (filtered out or not in this subset)
                    pass

            logger.info(f"Mapped to {len(sensitive_item_ids)} sensitive internal item IDs (out of {len(sensitive_asins)} ASINs)")

            # Evaluate all users in the dataset
            uid_field = dataset.uid_field
            user_ids_data = dataset.inter_feat[uid_field]
            # Handle both tensor and numpy array cases
            if isinstance(user_ids_data, torch.Tensor):
                user_ids = user_ids_data.cpu().numpy()
            else:
                user_ids = user_ids_data
            unique_user_ids = np.unique(user_ids).tolist()
            logger.info(f"\nEvaluating all {len(unique_user_ids)} users in the dataset")

            # Get predictions for all users
            max_k = 100  # Maximum k value for evaluation
            user_predictions = {}

            logger.info(f"Getting top-{max_k} predictions for {len(unique_user_ids)} users...")
            trainer.model.eval()
            with torch.no_grad():
                for user_idx, user_id in enumerate(unique_user_ids):
                    if (user_idx + 1) % 1000 == 0:
                        logger.info(f"  Processed {user_idx + 1}/{len(unique_user_ids)} users")

                    try:
                        # Create interaction for this user based on task type
                        if config['task_type'] == 'SBR':
                            # For sequential models, we need to provide item sequence
                            # Get the user's interaction data from the dataset
                            user_mask = dataset.inter_feat[dataset.uid_field] == user_id
                            user_indices = torch.where(user_mask)[0]

                            if len(user_indices) == 0:
                                # User has no interactions, skip
                                continue

                            # Get the last interaction which contains the longest/complete sequence
                            last_idx = user_indices[-1].item()

                            # Get the item sequence fields
                            item_seq_field = trainer.model.ITEM_SEQ
                            item_seq_len_field = trainer.model.ITEM_SEQ_LEN

                            interaction = {
                                dataset.uid_field: torch.tensor([user_id], device=trainer.device),
                                item_seq_field: dataset.inter_feat[item_seq_field][last_idx].unsqueeze(0).to(trainer.device),
                                item_seq_len_field: dataset.inter_feat[item_seq_len_field][last_idx].unsqueeze(0).to(trainer.device)
                            }
                        else:
                            # For CF models, only user_id is needed
                            interaction = {
                                'user_id': torch.tensor([user_id], device=trainer.device)
                            }

                        # Get predictions
                        scores = trainer.model.full_sort_predict(interaction)

                        # Get top-max_k items
                        _, topk_items = torch.topk(scores, k=max_k, dim=-1)
                        # Handle both 1D and 2D tensor outputs
                        topk_items_np = topk_items.cpu().numpy()
                        if topk_items_np.ndim > 1:
                            topk_items_np = topk_items_np[0]
                        user_predictions[user_id] = topk_items_np

                    except Exception as e:
                        logger.warning(f"Failed to get predictions for user {user_id}: {e}")
                        continue

            logger.info(f"Completed predictions for {len(user_predictions)} users")

            # Calculate metrics for different k values
            k_values = [5, 10, 20, 50, 100]
            sensitive_eval_results = {}

            for k in k_values:
                users_with_sensitive = 0
                total_sensitive_items = 0
                sensitive_counts = []

                for user_id, topk_items in user_predictions.items():
                    # Get top-k items for this k value
                    current_topk = topk_items[:k]

                    # Count sensitive items in top-k
                    sensitive_in_topk = sum(1 for item in current_topk if item in sensitive_item_ids)

                    if sensitive_in_topk > 0:
                        users_with_sensitive += 1
                        total_sensitive_items += sensitive_in_topk

                    sensitive_counts.append(sensitive_in_topk)

                # Calculate percentages and averages
                pct_users_with_sensitive = (users_with_sensitive / len(user_predictions) * 100) if user_predictions else 0
                avg_sensitive_per_user = (total_sensitive_items / len(user_predictions)) if user_predictions else 0
                avg_sensitive_per_affected_user = (total_sensitive_items / users_with_sensitive) if users_with_sensitive > 0 else 0

                sensitive_eval_results[f"k={k}"] = {
                    "users_with_sensitive": users_with_sensitive,
                    "total_users": len(user_predictions),
                    "pct_users_with_sensitive": pct_users_with_sensitive,
                    "total_sensitive_items": total_sensitive_items,
                    "avg_sensitive_per_user": avg_sensitive_per_user,
                    "avg_sensitive_per_affected_user": avg_sensitive_per_affected_user,
                    "min_sensitive": min(sensitive_counts) if sensitive_counts else 0,
                    "max_sensitive": max(sensitive_counts) if sensitive_counts else 0,
                }

                logger.info(f"\nTop-{k} Sensitive Item Metrics:")
                logger.info(f"  Users with sensitive items: {users_with_sensitive}/{len(user_predictions)} ({pct_users_with_sensitive:.2f}%)")
                logger.info(f"  Total sensitive items: {total_sensitive_items}")
                logger.info(f"  Avg sensitive items per user: {avg_sensitive_per_user:.4f}")
                logger.info(f"  Avg sensitive items per affected user: {avg_sensitive_per_affected_user:.4f}")
                logger.info(f"  Min/Max sensitive items: {min(sensitive_counts)}/{max(sensitive_counts)}")

            result["sensitive_item_evaluation"] = sensitive_eval_results
        else:
            logger.warning(f"Sensitive items file not found: {sensitive_items_path}")

    logger.info("")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate all original models for a given dataset")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    parser.add_argument("--checkpoint_dir", type=str, default="saved", help="Directory containing model checkpoints")
    parser.add_argument("--config_dir", type=str, default=".", help="Directory containing config files")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--models", "-m", type=str, nargs="+", default=None,
                       help="Filter models by name(s). If not specified, evaluates all models.")
    parser.add_argument("--sensitive_category", type=str, default=None,
                       help="Sensitive category for evaluation (e.g., 'health', 'alcohol')")
    parser.add_argument("--output_file", "-o", type=str, default=None,
                       help="Output file to save results (JSON format)")
    parser.add_argument("--task_type", type=str, default="CF",
                       help="Task type: 'CF' for collaborative filtering, 'SBR' for session-based rec (default: CF)")

    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    # Find all original models for this dataset
    print(f"Finding original models for dataset: {args.dataset}")
    if args.models:
        print(f"Filtering by models: {args.models}")
    models = find_original_models_for_dataset(args.dataset, args.checkpoint_dir, args.models)

    if not models:
        print(f"No original models (*_best.pth) found for dataset {args.dataset}")
        return

    print(f"Found {len(models)} original models for dataset {args.dataset}")
    print(f"Models: {[(m['model'], m['seed']) for m in models]}")
    print()

    # Get the first model to load the dataset
    first_model = models[0]
    model_lower = first_model["model"].lower()
    config_file = os.path.join(args.config_dir, f"config_{model_lower}.yaml")
    config_file_list = [config_file] if os.path.exists(config_file) else None

    config_dict = {
        "task_type": args.task_type,
        "gpu_id": int(args.cuda_visible_devices.split(",")[0]),
    }

    # Load dataset once
    print(f"Loading dataset: {args.dataset} (this will be reused for all models)")
    print(f"Task type: {args.task_type}")
    config, dataset, train_data, valid_data, test_data, logger = load_dataset_once(
        args.dataset,
        first_model["model"],
        config_file_list=config_file_list,
        config_dict=config_dict,
        sensitive_category=args.sensitive_category,
        task_type=args.task_type,
    )
    print(f"Dataset loaded successfully: {dataset}")
    print()

    # Evaluate each model
    all_results = []
    for i, model_info in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] Evaluating: {os.path.basename(model_info['model_file'])}")
        print(f"{'='*80}\n")

        try:
            result = evaluate_model(
                model_info,
                config,
                dataset,
                train_data,
                valid_data,
                test_data,
                logger,
                cuda_device=args.cuda_visible_devices.split(",")[0],
            )

            if result:
                all_results.append(result)
                print(f"\n✓ Completed: {model_info['model']} (seed={model_info['seed']})")
            else:
                print(f"\n✗ Failed: {model_info['model']} (seed={model_info['seed']})")
        except Exception as e:
            print(f"\n✗ Error evaluating {model_info['model']} (seed={model_info['seed']}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Finished evaluating {len(all_results)}/{len(models)} models for dataset {args.dataset}")
    print(f"{'='*80}\n")

    # Save results to file if requested
    if args.output_file:
        import json

        # Convert results to JSON-serializable format
        json_results = []
        for result in all_results:
            json_result = {
                "model": result["model"],
                "seed": result["seed"],
                "dataset": result["dataset"],
                "model_file": result["model_file"],
                "test_result": {k: float(v) for k, v in result["test_result"].items()},
            }
            if "sensitive_item_evaluation" in result:
                json_result["sensitive_item_evaluation"] = result["sensitive_item_evaluation"]
            json_results.append(json_result)

        with open(args.output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {args.output_file}")

    # Print summary
    print("\nSummary of Results:")
    print(f"{'Model':<15} {'Seed':<6} {'Test Metrics'}")
    print("-" * 80)
    for result in all_results:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in list(result["test_result"].items())[:3]])
        print(f"{result['model']:<15} {result['seed']:<6} {metrics_str}")


if __name__ == "__main__":
    main()
