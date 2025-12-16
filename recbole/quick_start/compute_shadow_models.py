"""
Shadow Model Computation Pipeline for RULI Privacy Evaluation

This module provides functionality to compute and save shadow models independently
from the evaluation pipeline. Shadow models are trained on different data partitions
and can be reused across multiple evaluation runs.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.quick_start.quick_start import k_subsets_exact_np


logger = getLogger()


def compute_shadow_models(
    model: str,
    dataset: str,
    config_file_list: Optional[List[str]] = None,
    config_dict: Optional[Dict] = None,
    k: int = 8,
    model_dir: str = "./models",
    gpu_id: Optional[int] = None,
    saved: bool = True,
    sensitive_category: Optional[str] = None,
    unlearning_fraction: Optional[float] = None,
    unlearning_sample_selection_seed: Optional[int] = None,
    task_type: str = "CF",
):
    """
    Compute and save shadow models for RULI Privacy evaluation.
    
    Shadow models are trained on the retain set (dataset excluding forget set),
    then partitioned using k-subsets for MIA evaluation.
    
    Args:
        model: Model name
        dataset: Dataset name
        config_file_list: List of config file paths
        config_dict: Configuration dictionary
        k: Number of shadow models to create (must be even)
        model_dir: Directory to save shadow models
        gpu_id: GPU ID to use
        saved: Whether to save models
        sensitive_category: Sensitive category for forget set (if applicable)
        unlearning_fraction: Unlearning fraction for forget set (if applicable)
        unlearning_sample_selection_seed: Seed for forget set selection (if applicable)
    
    Returns:
        List of paths to saved shadow models
    """
    if k % 2 != 0:
        raise ValueError(f"k (number of shadow models) must be even, got {k}")
    
    # Initialize config
    if config_dict is None:
        config_dict = {}
    
    config_dict.update({
        "model": model,
        "dataset": dataset,
        "task_type": task_type,
    })
    if gpu_id is not None:
        config_dict["gpu_id"] = gpu_id
    
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Computing {k} shadow models for {model} on {dataset}")
    
    # Load full dataset
    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)
    
    # Remove forget set if provided (shadow models should be trained on retain set)
    if sensitive_category is not None and unlearning_fraction is not None:
        import pandas as pd
        
        logger.info(f"Removing forget set (sensitive_category={sensitive_category}, "
                   f"unlearning_fraction={unlearning_fraction})")
        
        # Load forget set
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{dataset}_unlearn_pairs_sensitive_category_{sensitive_category}"
            f"_seed_{unlearning_sample_selection_seed or config['seed']}"
            f"_unlearning_fraction_{float(unlearning_fraction)}.inter"
        )
        
        if not os.path.exists(unlearning_samples_path):
            raise FileNotFoundError(
                f"Forget set not found: {unlearning_samples_path}. "
                f"Please generate forget set first."
            )
        
        # Load forget set based on task type
        uid_field = dataset_obj.uid_field
        iid_field = dataset_obj.iid_field
        
        if config["task_type"] == "CF":
            forget_set_df = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
                header=0,
            )
        elif config["task_type"] == "SBR":
            with open(unlearning_samples_path, 'r') as f:
                header_line = f.readline().strip()
            column_names = [col.split(':')[0] for col in header_line.split('\t')]
            forget_set_df = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=column_names,
                header=0,
            )
        else:
            raise ValueError(f"Unsupported task_type for shadow models: {config['task_type']}")
        
        logger.info(f"Loaded forget set with {len(forget_set_df)} interactions")
        
        # Create lookup set for forget interactions
        # Get user_ids and item_ids from dataset (these are already internal IDs)
        user_ids = dataset_obj.inter_feat[uid_field].to_numpy()
        item_ids = dataset_obj.inter_feat[iid_field].to_numpy()
        
        forget_pairs = set()
        for _, row in forget_set_df.iterrows():
            try:
                user_token = str(row[uid_field])
                item_token = str(row[iid_field])
                user_id = dataset_obj.token2id(uid_field, user_token)
                item_id = dataset_obj.token2id(iid_field, item_token)
                # Store as tuples with consistent types
                forget_pairs.add((int(user_id), int(item_id)))
            except (ValueError, KeyError):
                pass
        
        logger.info(f"Created forget_pairs set with {len(forget_pairs)} unique (user_id, item_id) pairs")
        
        # Remove forget set interactions from dataset
        # Create boolean mask: True for interactions to keep (not in forget set)
        user_ids_int = user_ids.astype(int)
        item_ids_int = item_ids.astype(int)
        
        # Create boolean mask: True for interactions to keep (not in forget set)
        removed_mask = np.array([
            (int(user_ids_int[i]), int(item_ids_int[i])) not in forget_pairs
            for i in range(len(user_ids_int))
        ])
        
        n_forget_removed = (~removed_mask).sum()
        dataset_obj = dataset_obj.copy(dataset_obj.inter_feat[removed_mask])
        logger.info(f"Removed {n_forget_removed} forget interactions (out of {len(forget_pairs)} in forget set). "
                   f"Retain set has {len(dataset_obj.inter_feat)} interactions")
    
    # Get unique users and create k subsets
    uid_field = dataset_obj.uid_field
    user_ids = dataset_obj.inter_feat[uid_field].to_numpy()
    unique_users = np.sort(np.unique(user_ids))
    
    logger.info(f"Total unique users: {len(unique_users)}")
    user_subsets = k_subsets_exact_np(unique_users, k=k)
    logger.info(f"Created {k} user subsets using k_subsets_exact_np")
    
    # Create shadow model directory under ./saved/shadow_models/
    shadow_models_dir = os.path.join("saved", "shadow_models")
    os.makedirs(shadow_models_dir, exist_ok=True)
    
    saved_models = []
    metadata = {
        "model": model,
        "dataset": dataset,
        "seed": config["seed"],
        "k": k,
        "partitions": []
    }
    
    # Train shadow models for each partition
    for partition_idx in range(k):
        logger.info(f"Training shadow model {partition_idx + 1}/{k}")
        
        # Create dataset excluding users in this partition (OUT model)
        users_to_drop = user_subsets[partition_idx]  # Already a list from k_subsets_exact_np
        removed_mask = ~np.isin(user_ids, users_to_drop)
        
        partition_dataset = dataset_obj.copy(dataset_obj.inter_feat[removed_mask])
        logger.info(f"Partition {partition_idx}: {len(users_to_drop)} users excluded, "
                   f"{len(partition_dataset.inter_feat)} interactions remaining")
        
        # Prepare data with same split as main experiment
        train_data, valid_data, test_data = data_preparation(config, partition_dataset)
        
        # Initialize model
        init_seed(config["seed"] + partition_idx, config["reproducibility"])
        shadow_model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
        
        # Initialize trainer
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, shadow_model)
        
        # Train model
        logger.info(f"Training shadow model partition {partition_idx}...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=True, saved=saved, show_progress=False
        )
        
        # Save model
        if saved:
            model_filename = (
                f"shadow_model_{model}_seed_{config['seed']}_dataset_{dataset}_"
                f"partition_{partition_idx}.pth"
            )
            model_path = os.path.join(shadow_models_dir, model_filename)
            
            # Save checkpoint
            checkpoint = {
                "state_dict": shadow_model.state_dict(),
                "other_parameter": shadow_model.get_other_parameter(),
                "config": config,
                "partition_idx": partition_idx,
            }
            torch.save(checkpoint, model_path)
            saved_models.append(model_path)
            logger.info(f"Saved shadow model to {model_path}")
            
            # Store metadata
            metadata["partitions"].append({
                "partition_idx": partition_idx,
                "excluded_users": users_to_drop,
                "n_excluded_users": len(users_to_drop),
                "model_path": model_path,
                "best_valid_score": best_valid_score,
            })
    
    # Save metadata
    metadata_path = None
    if saved:
        metadata_path = os.path.join(
            shadow_models_dir,
            f"shadow_models_metadata_seed_{config['seed']}.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved shadow models metadata to {metadata_path}")
        metadata["metadata_path"] = metadata_path
    
    logger.info(f"Completed computing {k} shadow models")
    return saved_models, metadata

