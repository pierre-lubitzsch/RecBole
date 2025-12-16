"""
RULI Privacy Evaluator for Game 2 MIA Evaluation

This module implements the RULI Privacy Evaluator based on Game 2 from the
usenixsecurity25-naderloui paper for Membership Inference Attack (MIA) evaluation
of unlearning.
"""

import os
import json
import numpy as np
import torch
import random
from typing import List, Dict, Optional, Tuple
from logging import getLogger
import pandas as pd

import torch.nn.functional as F


logger = getLogger()


class RULIPrivacyEvaluator:
    """
    RULI Privacy Evaluator for Game 2 MIA evaluation of unlearning.
    
    This evaluator uses shadow models to perform membership inference attacks
    on unlearned models, following the Game 2 methodology from the paper.
    """
    
    def __init__(
        self,
        config,
        dataset,
        train_data,
        test_data,
        shadow_models_dir: Optional[str] = None,
        k: int = 8,
        beta_threshold: float = 0.5,
        n_population_samples: int = 2500,
    ):
        """
        Initialize RULI Privacy Evaluator.
        
        Args:
            config: Configuration object
            dataset: Dataset object
            train_data: Training data loader
            test_data: Test data loader
            shadow_models_dir: Directory containing shadow models (default: {model_dir}/shadow_models/)
            k: Number of shadow models (must match shadow model computation)
            beta_threshold: Decision threshold for membership inference
            n_population_samples: Number of population samples for calibration
        """
        self.config = config
        self.dataset = dataset
        self.train_data = train_data
        self.test_data = test_data
        self.k = k
        self.beta_threshold = beta_threshold
        self.n_population_samples = n_population_samples
        
        # Determine shadow models directory
        if shadow_models_dir is None:
            shadow_models_dir = os.path.join("saved", "shadow_models")
        self.shadow_models_dir = shadow_models_dir
        
        # Load shadow models
        self.shadow_models = self._load_shadow_models()
        logger.info(f"Loaded {len(self.shadow_models)} shadow models from {shadow_models_dir}")
        
        # Store field names
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
    
    def _load_shadow_models(self) -> List[torch.nn.Module]:
        """Load pre-computed shadow models from disk."""
        shadow_models = []
        
        # Try to load metadata first
        metadata_path = os.path.join(
            self.shadow_models_dir,
            f"shadow_models_metadata_seed_{self.config['seed']}.json"
        )
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for partition_info in metadata["partitions"]:
                model_path = partition_info["model_path"]
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.config["device"])
                    
                    # Recreate model
                    from recbole.utils import get_model
                    shadow_model = get_model(self.config["model"])(
                        self.config, self.train_data._dataset
                    ).to(self.config["device"])
                    shadow_model.load_state_dict(checkpoint["state_dict"])
                    shadow_model.load_other_parameter(checkpoint.get("other_parameter"))
                    shadow_model.eval()
                    shadow_models.append(shadow_model)
                else:
                    logger.warning(f"Shadow model not found: {model_path}")
        else:
            # Fallback: try to load models directly by naming convention
            logger.warning(f"Metadata not found at {metadata_path}, trying direct loading")
            for partition_idx in range(self.k):
                model_filename = (
                    f"shadow_model_{self.config['model']}_seed_{self.config['seed']}_"
                    f"dataset_{self.config['dataset']}_partition_{partition_idx}.pth"
                )
                model_path = os.path.join(self.shadow_models_dir, model_filename)
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.config["device"])
                    
                    from recbole.utils import get_model
                    shadow_model = get_model(self.config["model"])(
                        self.config, self.train_data._dataset
                    ).to(self.config["device"])
                    shadow_model.load_state_dict(checkpoint["state_dict"])
                    shadow_model.load_other_parameter(checkpoint.get("other_parameter"))
                    shadow_model.eval()
                    shadow_models.append(shadow_model)
                else:
                    logger.warning(f"Shadow model not found: {model_path}")
        
        if len(shadow_models) == 0:
            raise FileNotFoundError(
                f"No shadow models found in {self.shadow_models_dir}. "
                f"Please run shadow model computation first."
            )
        
        if len(shadow_models) != self.k:
            logger.warning(
                f"Expected {self.k} shadow models, but loaded {len(shadow_models)}. "
                f"Proceeding with available models."
            )
        
        return shadow_models
    
    def construct_d_target(
        self,
        sensitive_category: str,
        forget_set: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Dict], List[bool]]:
        """
        Construct D_target from sensitive interactions.
        
        D_target can include samples from the forget set, but overlaps are tracked
        to avoid double-processing during unlearning.
        
        Args:
            sensitive_category: Sensitive category name
            forget_set: DataFrame containing forget set (D_forget) interactions
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (D_target interactions, overlap flags indicating which are in D_forget)
        """
        # Set seed for reproducibility
        old_np_state = None
        old_random_state = None
        if seed is not None:
            old_np_state = np.random.get_state()
            old_random_state = random.get_state()
            np.random.seed(seed)
            random.seed(seed)
        
        try:
            # Load sensitive items
            sensitive_items_path = os.path.join(
                self.config["data_path"],
                f"sensitive_asins_{sensitive_category}.txt"
            )
            
            if not os.path.exists(sensitive_items_path):
                sensitive_items_path = os.path.join(
                    self.config["data_path"],
                    f"sensitive_products_{sensitive_category}.txt"
                )
            
            if not os.path.exists(sensitive_items_path):
                raise FileNotFoundError(
                    f"Sensitive items file not found for category '{sensitive_category}'. "
                    f"Tried:\n  - sensitive_asins_{sensitive_category}.txt\n  "
                    f"- sensitive_products_{sensitive_category}.txt"
                )
            
            with open(sensitive_items_path, 'r') as f:
                sensitive_items_raw = [line.strip() for line in f if line.strip()]
            
            # Map to internal item IDs
            sensitive_item_ids = set()
            for item_token in sensitive_items_raw:
                try:
                    item_id = self.dataset.token2id(self.iid_field, item_token)
                    sensitive_item_ids.add(item_id)
                except (ValueError, KeyError):
                    pass
            
            logger.info(
                f"Loaded {len(sensitive_item_ids)} sensitive items "
                f"(out of {len(sensitive_items_raw)} raw items)"
            )
            
            # Find all sensitive interactions in full dataset
            user_ids = self.dataset.inter_feat[self.uid_field].to_numpy()
            item_ids = self.dataset.inter_feat[self.iid_field].to_numpy()
            
            sensitive_mask = np.isin(item_ids, list(sensitive_item_ids))
            sensitive_interactions = []
            
            for idx in np.where(sensitive_mask)[0]:
                interaction = {
                    "user_id": int(user_ids[idx]),
                    "item_id": int(item_ids[idx]),
                    "index": int(idx),
                }
                # Add other fields if present
                if "rating" in self.dataset.inter_feat.columns:
                    interaction["rating"] = float(self.dataset.inter_feat["rating"].iloc[idx])
                if "timestamp" in self.dataset.inter_feat.columns:
                    interaction["timestamp"] = float(self.dataset.inter_feat["timestamp"].iloc[idx])
                
                sensitive_interactions.append(interaction)
            
            logger.info(f"Found {len(sensitive_interactions)} sensitive interactions")
            
            # Create forget set lookup for overlap tracking
            forget_set_lookup = set()
            if forget_set is not None:
                for _, row in forget_set.iterrows():
                    user_token = str(row[self.uid_field])
                    item_token = str(row[self.iid_field])
                    try:
                        user_id = self.dataset.token2id(self.uid_field, user_token)
                        item_id = self.dataset.token2id(self.iid_field, item_token)
                        forget_set_lookup.add((user_id, item_id))
                    except (ValueError, KeyError):
                        pass
            
            # Split sensitive interactions by train/test split
            train_interactions = []
            test_interactions = []
            
            # Determine which interactions are in train vs test
            train_user_item_pairs = set()
            test_user_item_pairs = set()
            
            # Get train pairs
            for batch in self.train_data:
                if isinstance(batch, dict):
                    batch_user_ids = batch[self.uid_field].cpu().numpy()
                    batch_item_ids = batch[self.iid_field].cpu().numpy()
                    for u, i in zip(batch_user_ids, batch_item_ids):
                        train_user_item_pairs.add((int(u), int(i)))
            
            # Get test pairs
            for batch in self.test_data:
                if isinstance(batch, dict):
                    batch_user_ids = batch[self.uid_field].cpu().numpy()
                    batch_item_ids = batch[self.iid_field].cpu().numpy()
                    for u, i in zip(batch_user_ids, batch_item_ids):
                        test_user_item_pairs.add((int(u), int(i)))
            
            # Categorize sensitive interactions
            for interaction in sensitive_interactions:
                pair = (interaction["user_id"], interaction["item_id"])
                if pair in train_user_item_pairs:
                    train_interactions.append(interaction)
                elif pair in test_user_item_pairs:
                    test_interactions.append(interaction)
                # If not in either, we'll include it but note it's not in standard splits
            
            logger.info(
                f"Split sensitive interactions: {len(train_interactions)} train, "
                f"{len(test_interactions)} test"
            )
            
            # Sample from both train and test to construct D_target
            # Ensure balanced representation
            min_samples = min(len(train_interactions), len(test_interactions))
            if min_samples > 0:
                # Sample equal amounts from both
                n_samples_per_split = min(min_samples, self.n_population_samples // 2)
                
                train_samples = random.sample(
                    train_interactions,
                    min(n_samples_per_split, len(train_interactions))
                )
                test_samples = random.sample(
                    test_interactions,
                    min(n_samples_per_split, len(test_interactions))
                )
                
                d_target = train_samples + test_samples
            else:
                # If one split is empty, use all from the other
                d_target = train_interactions if len(train_interactions) > 0 else test_interactions
                d_target = random.sample(
                    d_target,
                    min(len(d_target), self.n_population_samples)
                )
            
            # Mark overlaps with forget set
            overlap_flags = []
            for interaction in d_target:
                pair = (interaction["user_id"], interaction["item_id"])
                is_in_forget = pair in forget_set_lookup
                overlap_flags.append(is_in_forget)
            
            n_overlaps = sum(overlap_flags)
            logger.info(
                f"Constructed D_target with {len(d_target)} samples "
                f"({len(d_target) - n_overlaps} not in forget set, {n_overlaps} in forget set)"
            )
            
            return d_target, overlap_flags
            
        finally:
            # Restore random state
            if seed is not None:
                if old_np_state is not None:
                    np.random.set_state(old_np_state)
                if old_random_state is not None:
                    random.setstate(old_random_state)
    
    def compute_mia_score(
        self,
        target_model: torch.nn.Module,
        target_sample: Dict,
        population_samples: List[Dict],
    ) -> float:
        """
        Compute MIA score for a sample using Game 2 methodology.
        
        Note: This is a placeholder implementation. The exact formula should be
        verified against the usenixsecurity25-naderloui paper.
        
        Args:
            target_model: Model to test (unlearned or retrained)
            target_sample: Sample to test membership for
            population_samples: Population dataset for calibration
        
        Returns:
            MIA score (higher = more likely to be a member)
        """
        # TODO: Implement exact Game 2 methodology from paper
        # This is a simplified version based on RMIA-like approach
        
        target_model.eval()
        
        # Compute probability of target sample under target model
        with torch.no_grad():
            # Create interaction tensor for target sample
            interaction = self._create_interaction(target_sample)
            target_scores = target_model.full_sort_predict(interaction)
            target_probs = F.softmax(target_scores, dim=-1)
            target_item_id = target_sample["item_id"]
            target_prob = target_probs[0, target_item_id].item()
        
        # Compute average probability under shadow models
        shadow_probs = []
        for shadow_model in self.shadow_models:
            shadow_model.eval()
            with torch.no_grad():
                interaction = self._create_interaction(target_sample)
                shadow_scores = shadow_model.full_sort_predict(interaction)
                shadow_probs_tensor = F.softmax(shadow_scores, dim=-1)
                shadow_prob = shadow_probs_tensor[0, target_item_id].item()
                shadow_probs.append(shadow_prob)
        
        avg_shadow_prob = np.mean(shadow_probs) if shadow_probs else 1e-10
        
        # Compute ratio
        ratio = target_prob / (avg_shadow_prob + 1e-10)
        
        # Compare against population
        population_ratios = []
        for pop_sample in population_samples[:self.n_population_samples]:
            with torch.no_grad():
                pop_interaction = self._create_interaction(pop_sample)
                pop_scores = target_model.full_sort_predict(pop_interaction)
                pop_probs = F.softmax(pop_scores, dim=-1)
                pop_item_id = pop_sample["item_id"]
                pop_prob = pop_probs[0, pop_item_id].item()
            
            # Average shadow model probability for population sample
            pop_shadow_probs = []
            for shadow_model in self.shadow_models:
                shadow_model.eval()
                with torch.no_grad():
                    pop_interaction = self._create_interaction(pop_sample)
                    pop_shadow_scores = shadow_model.full_sort_predict(pop_interaction)
                    pop_shadow_probs_tensor = F.softmax(pop_shadow_scores, dim=-1)
                    pop_shadow_prob = pop_shadow_probs_tensor[0, pop_item_id].item()
                    pop_shadow_probs.append(pop_shadow_prob)
            
            avg_pop_shadow_prob = np.mean(pop_shadow_probs) if pop_shadow_probs else 1e-10
            pop_ratio = pop_prob / (avg_pop_shadow_prob + 1e-10)
            population_ratios.append(pop_ratio)
        
        # MIA score: fraction of population with lower ratio
        mia_score = np.mean([ratio > r for r in population_ratios])
        
        return mia_score
    
    def _create_interaction(self, sample: Dict) -> Dict:
        """Create interaction tensor from sample dictionary."""
        interaction = {}
        user_id = sample["user_id"]
        
        if self.config["task_type"] == "SBR":
            # For SBR, need item sequence
            user_mask = self.dataset.inter_feat[self.uid_field] == user_id
            user_indices = np.where(user_mask)[0]
            
            if len(user_indices) > 0:
                last_idx = user_indices[-1]
                # Check if dataset has item sequence fields
                if "item_id_list" in self.dataset.inter_feat.columns:
                    item_seq_field = "item_id_list"
                    item_seq_len_field = "item_length"
                    interaction[self.uid_field] = torch.tensor([user_id], device=self.config["device"])
                    item_seq = self.dataset.inter_feat[item_seq_field].iloc[last_idx]
                    if isinstance(item_seq, torch.Tensor):
                        interaction[item_seq_field] = item_seq.unsqueeze(0).to(self.config["device"])
                    else:
                        interaction[item_seq_field] = torch.tensor([item_seq], device=self.config["device"])
                    if item_seq_len_field in self.dataset.inter_feat.columns:
                        seq_len = self.dataset.inter_feat[item_seq_len_field].iloc[last_idx]
                        interaction[item_seq_len_field] = torch.tensor([seq_len], device=self.config["device"])
                else:
                    # Fallback: just user_id
                    interaction[self.uid_field] = torch.tensor([user_id], device=self.config["device"])
            else:
                interaction[self.uid_field] = torch.tensor([user_id], device=self.config["device"])
        else:
            # For CF/NBR, just user_id
            interaction[self.uid_field] = torch.tensor(
                [user_id], device=self.config["device"]
            )
        
        return interaction
    
    def evaluate_unlearning(
        self,
        unlearned_model: torch.nn.Module,
        retrained_model: torch.nn.Module,
        d_target: List[Dict],
        overlap_flags: List[bool],
        population_samples: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate unlearning quality using RULI Privacy (Game 2).
        
        Args:
            unlearned_model: Model after unlearning procedure
            retrained_model: Model retrained from scratch without unlearned data
            d_target: Target dataset for MIA evaluation
            overlap_flags: Flags indicating which samples in D_target overlap with D_forget
            population_samples: Population dataset for calibration (if None, uses test_data)
        
        Returns:
            Tuple of (detailed results, summary metrics)
        """
        if population_samples is None:
            # Use test data as population
            population_samples = []
            for batch in self.test_data:
                if isinstance(batch, dict):
                    batch_user_ids = batch[self.uid_field].cpu().numpy()
                    batch_item_ids = batch[self.iid_field].cpu().numpy()
                    for u, i in zip(batch_user_ids, batch_item_ids):
                        population_samples.append({
                            "user_id": int(u),
                            "item_id": int(i),
                        })
        
        results = {
            "unlearned_model": {
                "d_target_scores": [],
                "d_target_detected": [],
                "d_target_not_forget_scores": [],
                "d_target_not_forget_detected": [],
            },
            "retrained_model": {
                "d_target_scores": [],
                "d_target_detected": [],
                "d_target_not_forget_scores": [],
                "d_target_not_forget_detected": [],
            },
        }
        
        # Evaluate each sample in D_target
        for sample, is_in_forget in zip(d_target, overlap_flags):
            # Skip samples that are in forget set (to avoid double-processing)
            if is_in_forget:
                continue
            
            # Test on unlearned model
            unlearned_score = self.compute_mia_score(
                unlearned_model, sample, population_samples
            )
            results["unlearned_model"]["d_target_scores"].append(unlearned_score)
            results["unlearned_model"]["d_target_detected"].append(
                unlearned_score >= self.beta_threshold
            )
            results["unlearned_model"]["d_target_not_forget_scores"].append(unlearned_score)
            results["unlearned_model"]["d_target_not_forget_detected"].append(
                unlearned_score >= self.beta_threshold
            )
            
            # Test on retrained model (baseline)
            retrained_score = self.compute_mia_score(
                retrained_model, sample, population_samples
            )
            results["retrained_model"]["d_target_scores"].append(retrained_score)
            results["retrained_model"]["d_target_detected"].append(
                retrained_score >= self.beta_threshold
            )
            results["retrained_model"]["d_target_not_forget_scores"].append(retrained_score)
            results["retrained_model"]["d_target_not_forget_detected"].append(
                retrained_score >= self.beta_threshold
            )
        
        # Compute summary metrics
        summary = {
            "unlearned_model": {
                "mean_mia_score": np.mean(results["unlearned_model"]["d_target_not_forget_scores"]),
                "detection_rate": np.mean(results["unlearned_model"]["d_target_not_forget_detected"]),
            },
            "retrained_model": {
                "mean_mia_score": np.mean(results["retrained_model"]["d_target_not_forget_scores"]),
                "detection_rate": np.mean(results["retrained_model"]["d_target_not_forget_detected"]),
            },
            "comparison": {
                "score_difference": (
                    np.mean(results["unlearned_model"]["d_target_not_forget_scores"]) -
                    np.mean(results["retrained_model"]["d_target_not_forget_scores"])
                ),
                "detection_rate_difference": (
                    np.mean(results["unlearned_model"]["d_target_not_forget_detected"]) -
                    np.mean(results["retrained_model"]["d_target_not_forget_detected"])
                ),
            },
            "n_samples_evaluated": len(results["unlearned_model"]["d_target_not_forget_scores"]),
            "n_samples_in_forget_set": sum(overlap_flags),
        }
        
        return results, summary

