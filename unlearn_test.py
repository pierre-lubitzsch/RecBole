from recbole.quick_start import unlearn_recbole
from recbole.config import Config
import argparse

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

def main():
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
        default=0.00001,
        help="percentage of training data to unlearn"
    )
    parser.add_argument(
        "--unlearning_sample_selection_method",
        type=str,
        default="popular",
        choices=["random", "popular", "unpopular"],
        help="how should the forget set be chosen (what type of interactions appear there)"
    )
    parser.add_argument(
        "--unlearning_algorithm",
        type=str,
        default="scif",
        choices=["scif", "kookmin", "fanchuan"],
        help="what unlearning algorithm to use",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=None,
        help="if not None, scif update step clipping will be applied with this value",
    )
    parser.add_argument(
        "--kookmin_init_rate",
        type=float,
        default=0.01,
        help="fraction of params to re-initialize with kookmin unlearning",
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
        default=100,
        help="number of training epochs"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="gpu to run on",
    )

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        set_seed(args.seed)

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = {
        "dataset": args.dataset,
        "unlearning_fraction": args.unlearning_fraction,
        "unlearning_sample_selection_method": args.unlearning_sample_selection_method,
        "model": args.model,
        "n_target_items": args.n_target_items,
        "unlearn_sample_selection_seed": args.seed,
        "seed": args.seed,
        "spam": args.spam,
        "unlearning_algorithm": args.unlearning_algorithm,
        "epochs": args.epochs,
        "gpu_id": args.gpu_id,
    }

    if args.spam:
        base_model_path = f"./saved/model_{args.model}_seed_{args.seed}_dataset_{args.dataset}_unlearning_fraction_{args.unlearning_fraction}_n_target_items_{args.n_target_items}_best.pth"
    else:
        base_model_path = f"./saved/model_{args.model}_seed_{args.seed}_dataset_{args.dataset}_best.pth"

    unlearn_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
        saved=True,
        unlearning_algorithm=args.unlearning_algorithm,
        max_norm=args.max_norm,
        base_model_path=base_model_path,
        kookmin_init_rate=args.kookmin_init_rate,
        spam=args.spam,
    )

if __name__ == "__main__":
    main()