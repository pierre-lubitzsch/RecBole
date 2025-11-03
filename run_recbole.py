# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run

import numpy as np
import random
import torch

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed to set, when not None"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=None,
        help="percentage of training data to unlearn"
    )
    parser.add_argument(
        "--unlearning_sample_selection_method",
        type=str,
        default=None,
        help="how should the forget set be chosen (what type of interactions appear there). example: sensitive_category_alcohol"
    )
    parser.add_argument(
        "--retrain_checkpoint_idx_to_match",
        type=int,
        default=None,
        help="checkpoint index of the users to unlearn, when not None"
    )
    parser.add_argument(
        "--spam",
        action="store_true",
        help="flag if we are in the spam setting"
    )
    parser.add_argument(
        "--n_target_items",
        type=int,
        default=10,
        help="number of target items the attacker wants to boost in the spam setting"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of training epochs"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="gpu to run on",
    )
    parser.add_argument(
        "--retrain_flag",
        action="store_true",
        help="flag to indicate if retraining is needed after unlearning"
    )
    parser.add_argument(
        "--rmia_out_model_flag",
        action="store_true",
        help="if set we train OUT models for RMIA",
    )
    parser.add_argument(
        "--rmia_out_model_partition_idx",
        type=int,
        default=None,
        help="if --rmia_out_model_flag is set, we train OUT models for RMIA with this partition index",
    )
    parser.add_argument(
        "--rmia_out_model_k",
        type=int,
        default=8,
        help="number of reference models for RMIA, default is 8",
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        help="sensitive category to evaluate (e.g., 'alcohol', 'meat')",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="task type for the model (e.g., 'CF' for collaborative filtering)",
    )
    parser.add_argument(
        "--max_training_hours",
        type=float,
        default=None,
        help="maximum number of hours for training/retraining (will stop before next epoch if time budget exceeded)",
    )

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        set_seed(args.seed)

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    # If sensitive_category is set and unlearning_sample_selection_method is not,
    # automatically set it to "sensitive_category_{category}"
    unlearning_sample_selection_method = args.unlearning_sample_selection_method
    if args.sensitive_category is not None and unlearning_sample_selection_method is None:
        unlearning_sample_selection_method = f"sensitive_category_{args.sensitive_category}"

    config_dict = {
        "dataset": args.dataset,
        "unlearning_fraction": args.unlearning_fraction,
        "unlearning_sample_selection_method": unlearning_sample_selection_method,
        "model": args.model,
        "retrain_checkpoint_idx_to_match": args.retrain_checkpoint_idx_to_match,
        "spam": args.spam,
        "n_target_items": args.n_target_items,
        "unlearn_sample_selection_seed": args.seed,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "retrain_flag": args.retrain_flag,
        "rmia_out_model_flag": args.rmia_out_model_flag,
        "rmia_out_model_partition_idx": args.rmia_out_model_partition_idx,
        "rmia_out_model_k": args.rmia_out_model_k,
        "sensitive_category": args.sensitive_category,
    }
    if args.task_type is not None:
        config_dict["task_type"] = args.task_type
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.max_training_hours is not None:
        config_dict["max_training_hours"] = args.max_training_hours

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
        retrain_flag=(args.retrain_checkpoint_idx_to_match is not None),
        unlearning_fraction=args.unlearning_fraction,
        unlearning_sample_selection_method=unlearning_sample_selection_method,
        retrain_checkpoint_idx_to_match=args.retrain_checkpoint_idx_to_match,
        config_dict=config_dict,
        spam=args.spam,
        sensitive_category=args.sensitive_category,
    )
