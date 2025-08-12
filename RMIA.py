import numpy as np
import torch
from typing import List, Dict, Optional
import torch.nn.functional as F

from recbole.quick_start import k_subsets_exact_np

class RMIAUnlearningEvaluator:
    def __init__(self, 
                 gamma: float = 2.0,
                 scaling_factor_a: float = 1,  # From RMIA paper for ImageNet
                 online: bool = False,
                 dataset=None):
        """
        RMIA implementation following Algorithm 1 from the paper
        
        Args:
            gamma: Threshold for pairwise LR test (γ in paper)
            scaling_factor_a: Scaling factor for offline mode (a in equation 10)
            online: Whether to use online mode (requires training IN models)
        """
        self.gamma = gamma
        self.a = scaling_factor_a
        self.online = online

        uid_field = dataset.uid_field
        user_ids = dataset.inter_feat[uid_field].to_numpy()
        unique_users = np.unique(user_ids)
        
        self.user_subsets = k_subsets_exact_np(unique_users, 10)
    
    def compute_pr_x_given_theta(self, model, interactions):
        """
        Compute Pr(x|theta) for a session in session-based recommendation
        This is the probability of the session under the model
        """
        model.eval()
        with torch.no_grad():
            # For session-based rec: compute product of prediction probabilities
            log_prob = 0
            for interaction in interactions:                
                scores = model.full_sort_predict(interaction)  # Shape: [batch_size, n_items]
                probas = F.softmax(scores, dim=-1)
                
                # Get the target item from the interaction
                if model.ITEM_ID in interaction:
                    target_items = interaction[model.ITEM_ID]
                else:
                    # You need to have the target item somewhere in your interaction
                    raise KeyError("No target item found in interaction")
                
                # Get probabilities for the target items
                batch_indices = torch.arange(scores.shape[0])
                target_probs = probas[batch_indices, target_items]
                
                # Sum log probabilities
                log_prob += torch.log(target_probs + 1e-10).sum()
            
        return torch.exp(log_prob).item()
    
    def compute_mia_score(self,
                         target_model,           # theta in Algorithm 1
                         target_sample,          # x in Algorithm 1
                         reference_models,       # theta in Algorithm 1
                         population_dataset,     # Dataset to sample Z from, use validation or test set
                         n_population_samples=2500):  # |Z| size
        """
        Algorithm 1: MIA Score Computation with RMIA
        
        Args:
            target_model: The model to test (unlearned model in your case)
            target_sample: The session to test membership for
            reference_models: List of reference models (or just retrained model)
            population_dataset: Dataset to sample population from
            n_population_samples: Number of population samples to use
        
        Returns:
            MIA score (fraction of population dominated by target sample)
        """
        
        # Line 1: Randomly choose subset Z from population dataset
        if len(population_dataset) > n_population_samples:
            Z = np.random.choice(population_dataset, n_population_samples, replace=False)
        else:
            Z = population_dataset
        
        # Line 2: Initialize counter
        C = 0
        
        # Lines 3-14: Compute Pr(x)
        if self.online:
            # Lines 4-10: Online mode (train IN models)
            # This is expensive - need to train k models with target_sample included
            raise NotImplementedError("Online mode requires training new models per query")
        else:
            # Lines 12-13: Offline mode - use only OUT models
            # Line 12: Compute Pr(x)_OUT
            pr_x_out = 0
            for i, ref_model in enumerate(reference_models):
                # TODO: fix this by using only the k / 2 reference models which did not contain the current user. need to check how to get user id from interaction
                # if target_sample[model.uid_field] not in self.user_subsets[i]:
                #     pr_x_out += self.compute_pr_x_given_theta(ref_model, target_sample)
                pr_x_out += self.compute_pr_x_given_theta(ref_model, target_sample)
            pr_x_out = pr_x_out / len(reference_models) / 2 if reference_models else 1e-10
            
            # Line 13: Apply scaling formula from equation 10
            pr_x = 0.5 * ((1 + self.a) * pr_x_out + (1 - self.a))
        
        # Line 15: Compute Ratio_x
        pr_x_given_theta = self.compute_pr_x_given_theta(target_model, target_sample)
        ratio_x = pr_x_given_theta / (pr_x + 1e-10)
        
        # Lines 16-22: Loop through population samples
        for z in Z:
            # Line 17: Compute Pr(z) using reference models
            pr_z = 0
            # Use the same logic as in the rmia OUT model training script to get the partition of training data and therefore the reference models
            for ref_model in reference_models:
                pr_z += self.compute_pr_x_given_theta(ref_model, z)
            pr_z = pr_z / len(reference_models) if reference_models else 1e-10
            
            # Line 18: Compute Ratio_z
            pr_z_given_theta = self.compute_pr_x_given_theta(target_model, z)
            ratio_z = pr_z_given_theta / (pr_z + 1e-10)
            
            # Lines 19-21: Check if x dominates z
            if (ratio_x / (ratio_z + 1e-10)) > self.gamma:
                C += 1
        
        # Line 23: Return MIA score
        score_mia = C / len(Z)
        return score_mia
    
    def evaluate_unlearning(self,
                           unlearned_model,
                           retrained_model,
                           unlearned_sessions: List,
                           retained_sessions: List,
                           population_sessions: List,
                           reference_models: Optional[List] = None,
                           beta_threshold: float = 0.5):
        """
        Evaluate unlearning quality using RMIA
        
        Args:
            unlearned_model: Model after unlearning procedure
            retrained_model: Model retrained from scratch without unlearned data
            unlearned_sessions: Sessions that should be forgotten
            retained_sessions: Sessions that should be remembered
            population_sessions: General population for comparison
            reference_models: List of reference models (if None, uses retrained_model)
            beta_threshold: Decision threshold β for membership inference
        
        Returns:
            Results dictionary with membership detection rates
        """
        
        # Use retrained model as reference if no others provided
        if reference_models is None:
            reference_models = [retrained_model]
        
        results = {
            'unlearned_model': {
                'unlearned_scores': [],
                'unlearned_detected': [],
                'retained_scores': [],
                'retained_detected': []
            },
            'retrained_model': {
                'unlearned_scores': [],
                'unlearned_detected': [],
                'retained_scores': [],
                'retained_detected': []
            }
        }
        
        # Test unlearned sessions (should NOT be detected as members)
        for session in unlearned_sessions:
            # Test on unlearned model
            score = self.compute_mia_score(
                target_model=unlearned_model,
                target_sample=session,
                reference_models=reference_models,
                population_dataset=population_sessions
            )
            results['unlearned_model']['unlearned_scores'].append(score)
            results['unlearned_model']['unlearned_detected'].append(score >= beta_threshold)
            
            # Test on retrained model (baseline - should also not detect)
            score = self.compute_mia_score(
                target_model=retrained_model,
                target_sample=session,
                reference_models=reference_models,
                population_dataset=population_sessions
            )
            results['retrained_model']['unlearned_scores'].append(score)
            results['retrained_model']['unlearned_detected'].append(score >= beta_threshold)
        
        # Test retained sessions (SHOULD be detected as members)
        for session in retained_sessions:
            # Test on unlearned model
            score = self.compute_mia_score(
                target_model=unlearned_model,
                target_sample=session,
                reference_models=reference_models,
                population_dataset=population_sessions
            )
            results['unlearned_model']['retained_scores'].append(score)
            results['unlearned_model']['retained_detected'].append(score >= beta_threshold)
            
            # Test on retrained model
            score = self.compute_mia_score(
                target_model=retrained_model,
                target_sample=session,
                reference_models=reference_models,
                population_dataset=population_sessions
            )
            results['retrained_model']['retained_scores'].append(score)
            results['retrained_model']['retained_detected'].append(score >= beta_threshold)
        
        # Compute summary metrics
        summary = {
            'unlearning_effectiveness': {
                'unlearned_model_leak_rate': np.mean(results['unlearned_model']['unlearned_detected']),
                'retrained_baseline_leak_rate': np.mean(results['retrained_model']['unlearned_detected']),
                'difference': np.mean(results['unlearned_model']['unlearned_detected']) - 
                             np.mean(results['retrained_model']['unlearned_detected'])
            },
            'retention_effectiveness': {
                'unlearned_model_retention_rate': np.mean(results['unlearned_model']['retained_detected']),
                'retrained_model_retention_rate': np.mean(results['retrained_model']['retained_detected'])
            },
            'scores': {
                'unlearned_data_mean_score': np.mean(results['unlearned_model']['unlearned_scores']),
                'retained_data_mean_score': np.mean(results['unlearned_model']['retained_scores'])
            }
        }
        
        return results, summary

# Usage example following Algorithm 1 exactly:
evaluator = RMIAUnlearningEvaluator(
    gamma=2.0,           # γ parameter from paper
    scaling_factor_a=0.3, # Adjust based on your dataset (0.3 for CIFAR-10)
    online=False         # Use offline mode (more practical)
)

# If you have multiple reference models trained on different subsets:
# reference_models = [model1, model2, model3, ...]
# Otherwise, just use the retrained model:
reference_models = [retrained_model]

results, summary = evaluator.evaluate_unlearning(
    unlearned_model=your_unlearned_model,
    retrained_model=your_retrained_model,
    unlearned_sessions=unlearned_sessions,
    retained_sessions=retained_sessions,
    population_sessions=all_available_sessions,
    reference_models=reference_models,
    beta_threshold=0.5
)

# Interpretation:
# Good unlearning: unlearned_model_leak_rate ≈ retrained_baseline_leak_rate (both low)
# Good retention: unlearned_model_retention_rate ≈ retrained_model_retention_rate (both high)