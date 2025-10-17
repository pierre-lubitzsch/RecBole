# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
import collections
import json
import pickle
from pathlib import Path

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import pandas as pd
import os
import torch
import numpy as np
import gc
import time

def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
    retrain_flag=False,
    unlearning_fraction=None,
    unlearning_sample_selection_method=None,
    retrain_checkpoint_idx_to_match=None,
    spam=False,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
            retrain_flag=retrain_flag,
            unlearning_fraction=unlearning_fraction,
            unlearning_sample_selection_method=unlearning_sample_selection_method,
            retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,
            spam=spam,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def k_subsets_exact_np(user_list, k=8):
    """
    Efficiently create k subsets where each sample appears in exactly k/2 subsets.
    Optimized for when k << n.
    
    Args:
        samples: list or array of samples
        k: number of subsets (must be even)
    
    Returns:
        list of k subsets
    """

    user_list = np.asarray(user_list)

    old_state = np.random.get_state()
    np.random.seed(2)

    assert k % 2 == 0, f"k (number of total reference models) must be even. currently k = {k}"
    
    n = len(user_list)
    half_k = k // 2
        
    inclusion_matrix = np.zeros((n, k), dtype=bool)
    random_selections = np.random.rand(n, k).argsort(axis=1)[:, :half_k]
    
    rows = np.arange(n)[:, None]
    inclusion_matrix[rows, random_selections] = True
    
    subsets = [user_list[inclusion_matrix[:, j]].tolist() for j in range(k)]
    
    np.random.set_state(old_state)

    return subsets
    
    


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
    retrain_flag=False,
    unlearning_fraction=None,
    unlearning_sample_selection_method=None,
    retrain_checkpoint_idx_to_match=None,
    spam=False,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """

    total_start_time = time.time()

    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config, unlearning=False, spam=spam)

    print("original dataset")
    logger.info(dataset)

    if retrain_flag:
        # remove unlearned interactions
        if "spam" in config and config["spam"]:
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_spam_sessions_dataset_{config['dataset']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}_seed_{config['unlearn_sample_selection_seed']}.inter"
            )
        else:
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_unlearn_pairs_{config['unlearning_sample_selection_method']}"
                f"_seed_{config['unlearn_sample_selection_seed']}"
                f"_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
            )

        unlearning_samples = pd.read_csv(
            unlearning_samples_path,
            sep="\t",
            names=["user_id", "item_id", "timestamp"],
            header=0,
        )

        pairs_by_user = (
            unlearning_samples.groupby("user_id")["item_id"]
            .agg(list)
            .to_dict()
        )

        unlearning_checkpoints = [len(pairs_by_user) // 4, len(pairs_by_user) // 2, 3 * len(pairs_by_user) // 4, len(pairs_by_user) - 1]
        users_unlearned = unlearning_checkpoints[retrain_checkpoint_idx_to_match]
        removed_mask = np.zeros(len(dataset.inter_feat), dtype=bool)

        uid_field, iid_field = dataset.uid_field, dataset.iid_field
        user_ids = dataset.inter_feat[uid_field].to_numpy()
        item_ids = dataset.inter_feat[iid_field].to_numpy()

        pairs_by_user_unlearned = sorted(pairs_by_user.items())[:users_unlearned + 1]

        for unlearn_request_idx, (u, forget_items) in enumerate(pairs_by_user_unlearned):
            all_idx = np.where(user_ids == u)[0]
            mask = np.isin(item_ids[all_idx], forget_items)
            removed_mask[all_idx[~mask]] = True

        dataset = dataset.copy(dataset.inter_feat[~removed_mask])

        print("retain dataset")
        logger.info(dataset)
    
    elif config["rmia_out_model_flag"]:
        uid_field = dataset.uid_field
        user_ids = dataset.inter_feat[uid_field].to_numpy()
        unique_users = np.sort(np.unique(user_ids))
        
        user_subsets = k_subsets_exact_np(unique_users, k=config["rmia_out_model_k"])

        users_to_drop = user_subsets[config["rmia_out_model_partition_idx"]].tolist()
        rmia_removed_mask = ~np.isin(user_ids, list(users_to_drop))

        dataset = dataset.copy(dataset.inter_feat[rmia_removed_mask])

        print(f"rmia dataset for OUT model with idx: {config['rmia_out_model_partition_idx']}")
        logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset, spam=spam)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=saved,
        show_progress=config["show_progress"],
        retrain_flag=retrain_flag,
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process



def unlearn_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
    unlearning_algorithm="scif",
    max_norm=None,
    base_model_path=None,
    kookmin_init_rate=0.01,
    spam=False,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """

    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    retrain_checkpoint_idx_to_match = 0
    config["retrain_checkpoint_idx_to_match"] = retrain_checkpoint_idx_to_match
    # different batch_size for unlearning
    # TODO: make params
    config["train_batch_size"] = 256
    config["eval_batch_size"] = 512
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset = create_dataset(config, unlearning=False, spam=spam)
    # save orig_df for forget set creation because inter_feat gets converted to torch during dataset.build()
    orig_inter_df = dataset.inter_feat.copy()

    removed_mask = np.zeros(len(orig_inter_df), dtype=bool)
    logger.info(dataset)

    target_items = None
    if spam:
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{config['dataset']}_spam_sessions_dataset_{config['dataset']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}_seed_{config['unlearn_sample_selection_seed']}.inter"
        )
        unlearning_samples_metadata_path = os.path.join(
            config["data_path"],
            f"spam_metadata_dataset_{config['dataset']}_unlearning_fraction_{config['unlearning_fraction']}_n_target_items_{config['n_target_items']}_seed_{config['unlearn_sample_selection_seed']}.json"
        )
        with open(unlearning_samples_metadata_path, "r") as f:
            metadata = json.load(f)
            target_items = np.array(list(map(int, metadata["target_items"])), dtype=np.int64)
    else:
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{config['dataset']}_unlearn_pairs_{config['unlearning_sample_selection_method']}"
            f"_seed_{config['unlearn_sample_selection_seed']}"
            f"_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
        )

    print("loaded dataset")

    if config.task_type != "CF":
        unlearning_samples = pd.read_csv(
            unlearning_samples_path,
            sep="\t",
            names=["user_id", "item_id", "timestamp"],
            header=0,
        )
    else:
        unlearning_samples = pd.read_csv(
            unlearning_samples_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=0,
        )

    print("loaded unlearning samples")

    uid_field, iid_field = dataset.uid_field, dataset.iid_field

    user_ids = dataset.inter_feat[uid_field].to_numpy()
    item_ids = dataset.inter_feat[iid_field].to_numpy()

    print("created user and item ids")

    
    rows_by_user = dict()

    pairs_by_user = (
        unlearning_samples.groupby("user_id")["item_id"]
        .agg(list)
        .to_dict()
    )

    # model loading and initialization

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, dataset).to(config["device"])

    checkpoint = torch.load(base_model_path)

    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    logger.info(model)

    # trainer loading and initialization
    # scale down the lr for non-reset params (which are the majority) during training we will scale the lr for the reset params to the normal value
    if unlearning_algorithm == "kookmin":
        config["learning_rate"] *= 0.1
        
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    train_data, valid_data, test_data = data_preparation(
        config, dataset, spam=spam,
    )

    # sanity check
    # print("evaluation of original model:\n")
    # test_result = trainer.evaluate(
    #     test_data, load_best_model=True, show_progress=True, model_file=base_model_path, collect_target_probabilities=spam, target_items=target_items,
    # )
    # if spam:
    #     test_result, probability_data = test_result
    # print(test_result)

    # negative item sampler used for unlearning later which can sample unseen items for all users
    unlearning_sampler = train_data._sampler

    # Determine how many user sessions we need per request
    if unlearning_algorithm == "scif":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "kookmin":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "fanchuan":
        retain_sessions_per_request = 32  # Number of complete user sessions
        contrastive_samples_per_iteration = 8  # For contrastive learning iterations
    
    # Pre-sample retain data for all unlearning requests
    total_unlearn_requests = len(pairs_by_user)
    
    # Get all unique users from orig_inter_df
    unique_users = orig_inter_df['user_id'].unique()
    user_pool = unique_users.tolist()
    
    # Create a mapping of user to their session indices
    user_to_indices = orig_inter_df.groupby('user_id').indices
    
    # Create an oversized pool to account for filtering needs
    pool_size_multiplier = 3  # Make pool 3x larger to ensure we have enough replacements
    
    if unlearning_algorithm == "fanchuan":
        base_sessions_needed = (
            retain_sessions_per_request * total_unlearn_requests * 
            contrastive_samples_per_iteration
        )
    else:
        base_sessions_needed = retain_sessions_per_request * total_unlearn_requests
    
    total_pool_size = base_sessions_needed * pool_size_multiplier
    
    # Create the oversized pool of users
    user_sample_pool = []
    while len(user_sample_pool) < total_pool_size:
        shuffled_users = np.random.permutation(user_pool)
        user_sample_pool.extend(shuffled_users.tolist())
    
    # Track which pool indices we've used
    pool_cursor = 0
    
    def get_retain_sessions_excluding_unlearned_users(
        num_interactions_needed,
        unlearned_users_before,
        user_pool,
        cursor,
        user_to_indices,
    ):
        """
        Get a set of full user sessions until we have at least `num_interactions_needed` interactions,
        skipping unlearned users.
        """
        selected_indices = []
        selected_users = []
        current_cursor = cursor
        total_interactions = 0

        while total_interactions < num_interactions_needed:
            if current_cursor >= len(user_pool):
                current_cursor = 0
                np.random.shuffle(user_pool)

            user_id = user_pool[current_cursor]
            current_cursor += 1

            if user_id in unlearned_users_before:
                continue

            session_indices = user_to_indices[user_id]
            session_len = len(session_indices)

            selected_indices.extend(session_indices)
            selected_users.append(user_id)
            total_interactions += session_len

        return selected_indices, selected_users, current_cursor
    
    # Now in the unlearning loop:
    unlearned_users_before = []
    unlearning_checkpoints = [len(pairs_by_user) // 4, len(pairs_by_user) // 2, 3 * len(pairs_by_user) // 4, len(pairs_by_user) - 1]
    eval_files = [f"{trainer.saved_model_file[:-len('.pth')]}_unlearn_epoch_{e}_retrain_checkpoint_idx_to_match_{r}.pth" for e, r in zip(unlearning_checkpoints, range(4))]
    eval_masks = []
    
    unlearning_times = []
    total_start_time = time.time()
    
    for unlearn_request_idx, (u, forget_items) in enumerate(sorted(pairs_by_user.items())):
        print(f"\nUnlearning request {unlearn_request_idx + 1}/{len(pairs_by_user)} for user {u}\n")
        request_start_time = time.time()

        unlearned_users_before.append(u)

        rows_by_user[u] = np.where(user_ids == u)[0]
        all_idx = rows_by_user[u]
        mask = np.isin(item_ids[all_idx], forget_items)
        removed_mask[all_idx[mask]] = True

        saved_checkpoint = unlearn_request_idx in unlearning_checkpoints

        if saved_checkpoint:
            eval_mask = removed_mask.copy()
            eval_masks.append(eval_mask)

        if "eval_only" in config and config["eval_only"]:
            continue

        cur_forget_ds = dataset.copy(
            orig_inter_df.iloc[all_idx]
        )

        print(f"Unlearning {len(cur_forget_ds)} interactions")

        clean_forget_ds = dataset.copy(
            orig_inter_df.iloc[all_idx[~mask]]
        )

        forget_data = data_preparation(config, cur_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
        clean_forget_data = data_preparation(config, clean_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)

        retain_limit_absolute = int(0.1 * len(orig_inter_df))  # 10% of full dataset

        avg_session_length = len(orig_inter_df) / len(unique_users)

        if unlearning_algorithm == "scif":
            retain_batch_size = 16
            samples_wanted_constant = 1024
            retain_samples_used = 128
            forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
            
            total_samples_needed = max(
                retain_samples_used * forget_size,
                samples_wanted_constant * forget_size
            )

            # Cap to 10% of dataset
            total_samples_needed = min(total_samples_needed, retain_limit_absolute)
            sessions_needed = int(total_samples_needed / avg_session_length) + 1

        elif unlearning_algorithm == "kookmin":
            retain_batch_size = config["train_batch_size"]
            forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

            neg_grad_retain_sample_size = 128 * forget_size
            retain_samples_used_for_update = 32 * forget_size

            total_samples_needed = max(
                neg_grad_retain_sample_size,
                retain_samples_used_for_update
            )

            # Cap to 10% of dataset
            total_samples_needed = min(total_samples_needed, retain_limit_absolute)
            sessions_needed = int(total_samples_needed / avg_session_length) + 1

        elif unlearning_algorithm == "fanchuan":
            retain_batch_size = config["train_batch_size"]
            forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

            retain_samples_used_for_update = 32 * forget_size
            unlearn_iters_contrastive = 8

            total_samples_needed = retain_samples_used_for_update * unlearn_iters_contrastive

            # Cap to 10% of dataset
            total_samples_needed = min(total_samples_needed, retain_limit_absolute)
            sessions_needed = int(total_samples_needed / avg_session_length) + 1
    
        # Get complete sessions
        retain_indices, retain_users, pool_cursor = get_retain_sessions_excluding_unlearned_users(
            sessions_needed,
            unlearned_users_before,
            user_sample_pool,
            pool_cursor,
            user_to_indices,
        )
        
        print(f"Selected {len(retain_users)} user sessions with {len(retain_indices)} total interactions")
        
        # Create dataset from selected indices
        current_retain_data = orig_inter_df.iloc[retain_indices]
        current_retain_ds = dataset.copy(current_retain_data)
        
        # Temporarily modify batch size for this retain loader
        tmp = config["train_batch_size"]
        config["train_batch_size"] = retain_batch_size
        current_retain_loader = data_preparation(config, current_retain_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
        config["train_batch_size"] = tmp

        # model training
        trainer.unlearn(
            unlearn_request_idx,
            forget_data,
            clean_forget_data,
            retain_train_data=current_retain_loader,
            retain_valid_data=None,#valid_data,
            retain_test_data=None,#test_data,
            unlearning_algorithm=unlearning_algorithm,
            saved=saved_checkpoint,
            show_progress=False,  # no progress bar during unlearning as it is short either way
            max_norm=max_norm,
            unlearned_users_before=unlearned_users_before,
            kookmin_init_rate=kookmin_init_rate,
            retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,
            task_type=config.task_type,
        )

        request_end_time = time.time()
        request_time = request_end_time - request_start_time
        unlearning_times.append(request_time)

        print(f"\n\nRequest {unlearn_request_idx + 1} completed in {request_time:.2f} seconds\n\n")

        if saved_checkpoint:
            retrain_checkpoint_idx_to_match += 1
            config["retrain_checkpoint_idx_to_match"] = retrain_checkpoint_idx_to_match

        sys.stdout.flush()

        gc.collect()


    if "eval_only" not in config or not config["eval_only"]:
        total_end_time = time.time()
        total_unlearning_time = total_end_time - total_start_time

        print(f"\nUnlearning Time Summary")
        print(f"Total unlearning time: {total_unlearning_time:.2f} seconds ({total_unlearning_time/60:.2f} minutes)")
        print(f"Average time per request: {np.mean(unlearning_times):.2f} seconds")
        print(f"Min time per request: {np.min(unlearning_times):.2f} seconds")
        print(f"Max time per request: {np.max(unlearning_times):.2f} seconds")
        print(f"Total requests processed: {len(unlearning_times)}")

    # eval
    # set eval batch size manually for now, as there is no parameter yet. change this in the future
    config["eval_batch_size"] = 256
    results = []

    # Dictionary to store per-model interaction data as lists of tuples
    model_interaction_probabilities = {}

    print("Original Model:")
    print(f"Evaluating model {base_model_path} on data with current mask\n")
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=True, model_file=base_model_path, collect_target_probabilities=spam, target_items=target_items,
    )
    if spam:
        test_result, probability_data = test_result
    print(test_result)

    # First loop: evaluate each model on its corresponding masked data
    for i, (file, mask) in enumerate(zip(eval_files, eval_masks)):
        print(f"Evaluating model {file} on data with current mask\n")
        
        model_interaction_probabilities[file] = []
        
        # Clear GPU cache before each evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # test on data with just current poisoned data removed
        cur_eval_df = orig_inter_df.loc[~mask]
        cur_eval_dataset = dataset.copy(cur_eval_df)
        cur_train_data, cur_val_data, cur_test_data = data_preparation(config, cur_eval_dataset, spam=spam, sampler=unlearning_sampler)

        del cur_train_data, cur_val_data  # we only need test data for evaluation
        gc.collect()

        test_result = trainer.evaluate(
            cur_test_data, load_best_model=True, show_progress=False, model_file=file, collect_target_probabilities=spam, target_items=target_items,
        )
        if spam:
            test_result, probability_data = test_result
            model_interaction_probabilities[file] = probability_data

        result = {
            "test_result": test_result,
            "model_file": file,
            "mask_type": "current",
        }
        results.append(result)
        print(f"Results for model {file} only removing currently poisoned data: {test_result}")
        
        # clear the current test data
        del cur_eval_df, cur_eval_dataset, cur_test_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n")
        sys.stdout.flush()

    print("Creating unpoisoned dataset for all models...")
    unpoisoned_eval_df = orig_inter_df.loc[~eval_masks[-1]]
    unpoisoned_eval_dataset = dataset.copy(unpoisoned_eval_df)
    unpoisoned_train_data, unpoisoned_val_data, unpoisoned_test_data = data_preparation(config, unpoisoned_eval_dataset, spam=spam)

    del unpoisoned_train_data, unpoisoned_val_data  # we only need test data for evaluation
    gc.collect()

    # Second loop: evaluate all models on the same unpoisoned data
    for file in eval_files:
        print(f"Evaluating model {file} on unpoisoned data\n")
        
        unpoisoned_key = f"{file}_unpoisoned"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unpoisoned_test_result = trainer.evaluate(
            unpoisoned_test_data, load_best_model=True, show_progress=False, model_file=file, collect_target_probabilities=spam, target_items=target_items,
        )
        if spam:
            unpoisoned_test_result, probability_data = unpoisoned_test_result
            model_interaction_probabilities[unpoisoned_key] = probability_data

        unpoisoned_result = {
            "test_result": unpoisoned_test_result,
            "model_file": file,
            "mask_type": "unpoisoned",
        }
        results.append(unpoisoned_result)

        print(f"Results for model {file} on unpoisoned data: {unpoisoned_test_result}")
        print("\n")
        sys.stdout.flush()

    del unpoisoned_eval_df, unpoisoned_eval_dataset, unpoisoned_test_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    

    if spam:
        pickle_file = os.path.join(trainer.saved_model_file[:-len(".pth")], "model_interaction_probabilities.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(model_interaction_probabilities, f)
        print(f"Saved model interaction probabilities to {pickle_file}")

    return results

def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:3],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config, unlearning=False, spam=config["spam"])
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset, spam=config["spam"])

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
