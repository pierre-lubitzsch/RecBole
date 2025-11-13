#!/usr/bin/env python3
"""
Script to evaluate all models for a given dataset.
Loads the dataset once and evaluates all models for that dataset.
Determines evaluation type (normal/retraining/unlearning) from model filename.
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path

def parse_model_filename(filename):
    """Parse model filename to extract parameters and determine evaluation type.
    
    Returns:
        dict with keys: model, dataset, seed, eval_type, and other relevant params
        eval_type can be: 'normal', 'retraining', 'unlearning', 'spam', 'rmia'
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
            "eval_type": "normal"
        }
    
    # Pattern for retraining: model_{model}_seed_{seed}_dataset_{dataset}_retrain_checkpoint_idx_to_match_{idx}_unlearning_fraction_{fraction}_unlearning_sample_selection_method_{method}.pth
    retrain_pattern = r"model_(\w+)_seed_(\d+)_dataset_([\w_]+)_retrain_checkpoint_idx_to_match_(\d+)_unlearning_fraction_([\d.]+)_unlearning_sample_selection_method_([\w_]+)\.pth"
    match = re.match(retrain_pattern, basename)
    if match:
        return {
            "model": match.group(1),
            "seed": int(match.group(2)),
            "dataset": match.group(3),
            "retrain_checkpoint_idx_to_match": int(match.group(4)),
            "unlearning_fraction": float(match.group(5)),
            "unlearning_sample_selection_method": match.group(6),
            "eval_type": "retraining"
        }
    
    # Pattern for unlearning: model_{model}_seed_{seed}_dataset_{dataset}_unlearning_algorithm_{algorithm}_unlearning_fraction_{fraction}_unlearning_sample_selection_method_{method}.pth
    unlearn_pattern = r"model_(\w+)_seed_(\d+)_dataset_([\w_]+)_unlearning_algorithm_(\w+)_unlearning_fraction_([\d.]+)_unlearning_sample_selection_method_([\w_]+)\.pth"
    match = re.match(unlearn_pattern, basename)
    if match:
        return {
            "model": match.group(1),
            "seed": int(match.group(2)),
            "dataset": match.group(3),
            "unlearning_algorithm": match.group(4),
            "unlearning_fraction": float(match.group(5)),
            "unlearning_sample_selection_method": match.group(6),
            "eval_type": "unlearning"
        }
    
    # Pattern for spam: model_{model}_seed_{seed}_dataset_{dataset}_unlearning_fraction_{fraction}_n_target_items_{n}_best.pth
    spam_pattern = r"model_(\w+)_seed_(\d+)_dataset_([\w_]+)_unlearning_fraction_([\d.]+)_n_target_items_(\d+)_best\.pth"
    match = re.match(spam_pattern, basename)
    if match:
        return {
            "model": match.group(1),
            "seed": int(match.group(2)),
            "dataset": match.group(3),
            "unlearning_fraction": float(match.group(4)),
            "n_target_items": int(match.group(5)),
            "eval_type": "spam"
        }
    
    # Pattern for RMIA OUT models: model_{model}_seed_{seed}_dataset_{dataset}_rmia_out_model_partition_idx_{idx}.pth
    rmia_pattern = r"model_(\w+)_seed_(\d+)_dataset_([\w_]+)_rmia_out_model_partition_idx_(\d+)\.pth"
    match = re.match(rmia_pattern, basename)
    if match:
        return {
            "model": match.group(1),
            "seed": int(match.group(2)),
            "dataset": match.group(3),
            "rmia_out_model_partition_idx": int(match.group(4)),
            "eval_type": "rmia"
        }
    
    return None


def find_models_for_dataset(dataset, checkpoint_dir="saved"):
    """Find all model files for a given dataset."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []
    
    # Find all .pth files
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    # Filter by dataset and parse
    models = []
    for model_file in model_files:
        parsed = parse_model_filename(model_file)
        if parsed and parsed["dataset"] == dataset:
            parsed["model_file"] = model_file
            models.append(parsed)
    
    return models


def build_eval_command(model_info, config_dir="."):
    """Build the command to evaluate a model."""
    model = model_info["model"]
    dataset = model_info["dataset"]
    seed = model_info["seed"]
    model_lower = model.lower()
    config_file = os.path.join(config_dir, f"config_{model_lower}.yaml")
    
    cmd = [
        "python", "run_recbole.py",
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--config_files", config_file,
        "--eval_only"
    ]
    
    eval_type = model_info["eval_type"]
    
    if eval_type == "retraining":
        # Extract category from unlearning_sample_selection_method
        method = model_info["unlearning_sample_selection_method"]
        if method.startswith("sensitive_category_"):
            category = method.replace("sensitive_category_", "")
        else:
            category = ""
        
        cmd.extend([
            "--unlearning_fraction", str(model_info["unlearning_fraction"]),
            "--sensitive_category", category,
            "--retrain_checkpoint_idx_to_match", str(model_info["retrain_checkpoint_idx_to_match"]),
            "--task_type", "CF",
            "--retrain_flag",
            "--unlearning_sample_selection_method", method
        ])
    elif eval_type == "unlearning":
        cmd.extend([
            "--unlearning_fraction", str(model_info["unlearning_fraction"]),
            "--unlearning_sample_selection_method", model_info["unlearning_sample_selection_method"],
            "--task_type", "CF"
        ])
    elif eval_type == "spam":
        cmd.extend([
            "--unlearning_fraction", str(model_info["unlearning_fraction"]),
            "--n_target_items", str(model_info["n_target_items"]),
            "--spam",
            "--task_type", "CF"
        ])
    elif eval_type == "rmia":
        cmd.extend([
            "--rmia_out_model_flag",
            "--rmia_out_model_partition_idx", str(model_info["rmia_out_model_partition_idx"]),
            "--task_type", "CF"
        ])
    else:  # normal
        cmd.append("--task_type")
        cmd.append("CF")
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models for a given dataset")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    parser.add_argument("--checkpoint_dir", type=str, default="saved", help="Directory containing model checkpoints")
    parser.add_argument("--config_dir", type=str, default=".", help="Directory containing config files")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value (e.g., '0' or '1,2')")
    parser.add_argument("--log_dir", type=str, default="eval_logs", help="Directory to save evaluation logs")
    
    args = parser.parse_args()
    
    # Find all models for this dataset
    print(f"Finding models for dataset: {args.dataset}")
    models = find_models_for_dataset(args.dataset, args.checkpoint_dir)
    
    if not models:
        print(f"No models found for dataset {args.dataset}")
        return
    
    print(f"Found {len(models)} models for dataset {args.dataset}")
    print(f"Models: {[m['model'] for m in models]}")
    print()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Evaluate each model
    for i, model_info in enumerate(models, 1):
        model_file = model_info["model_file"]
        eval_type = model_info["eval_type"]
        
        print(f"[{i}/{len(models)}] Evaluating: {os.path.basename(model_file)}")
        print(f"  Model: {model_info['model']}, Seed: {model_info['seed']}, Type: {eval_type}")
        
        # Build command
        cmd = build_eval_command(model_info, args.config_dir)
        
        # Create log filename
        log_filename = os.path.basename(model_file).replace(".pth", ".log")
        log_filepath = os.path.join(args.log_dir, log_filename)
        
        print(f"  Command: CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} {' '.join(cmd)}")
        print(f"  Log: {log_filepath}")
        print()
        
        # Run evaluation
        import subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        shell_cmd = ' '.join(f"'{arg}'" if ' ' in arg else arg for arg in cmd)
        shell_cmd = f"{shell_cmd} 2>&1 | tee {log_filepath}"
        
        result = subprocess.run(
            shell_cmd,
            shell=True,
            env=env
        )
        
        if result.returncode == 0:
            print(f"  ✓ Completed: {os.path.basename(model_file)}")
        else:
            print(f"  ✗ Failed: {os.path.basename(model_file)} (return code: {result.returncode})")
        print()
    
    print(f"Finished evaluating {len(models)} models for dataset {args.dataset}")


if __name__ == "__main__":
    main()

