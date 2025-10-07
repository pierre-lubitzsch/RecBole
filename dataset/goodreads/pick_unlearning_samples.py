import pandas as pd
import numpy as np
import collections
import argparse
import random
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Process unlearning percentage.")
    parser.add_argument(
        "--unlearning_fractions", 
        type=float,
        nargs='+',
        default=[0.0001],
        help='Space-separated list of unlearning fractions (e.g., 0.00001 0.0001)'
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "popular", "unpopular"],
        default="random",
        help="how are the interactions picked"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="random seed"
    )
    
    return parser.parse_args()

def apply_recbole_ls_split(df, train_ratio=0.8):
    """
    Apply RecBole's LS splitting logic to get the training set.
    With valid_and_test=[0.8, 0.1, 0.1], we take the first 80% of each user's sequence.
    """
    train_rows = []
    
    # Group by user and sort by timestamp
    for user_id, group in df.groupby("user_id"):
        # Sort by timestamp
        user_interactions = group.sort_values("timestamp").reset_index(drop=True)
        n_interactions = len(user_interactions)
        
        # Calculate split point for training (80%)
        train_size = int(n_interactions * train_ratio)
        
        # Ensure at least 1 interaction in train if user has any interactions
        if train_size == 0 and n_interactions > 0:
            train_size = 1
            
        # Take first train_size interactions
        train_rows.append(user_interactions.iloc[:train_size])
    
    return pd.concat(train_rows, ignore_index=True)

def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load full dataset
    train_dataset_file = "goodreads.inter"
    df_full = pd.read_csv(train_dataset_file, sep="\t", header=0, 
                          names=["user_id", "item_id", "rating", "timestamp"])
    n_interactions_before_filtering = len(df_full)

    # Filter out sessions with only 1 item, as they are not used for training
    session_counts = df_full["user_id"].value_counts()
    valid_sessions = session_counts[session_counts >= 2].index
    df_full = df_full[df_full["user_id"].isin(valid_sessions)].reset_index(drop=True)
    
    print(f"length 1 session count: {n_interactions_before_filtering - len(df_full)}")
    print(f"interactions after filtering: {len(df_full)}")

    # Apply RecBole LS split to get training set only
    df_train = apply_recbole_ls_split(df_full, train_ratio=0.8)
    n_interactions = len(df_train)
    session_indices = df_train.groupby("user_id").indices
    
    print(f"training interactions after split: {n_interactions}")

    if args.method in ["popular", "unpopular"]:
        item_to_session_and_timestamp = collections.defaultdict(list)
        for row in df_train.itertuples(index=False):
            item_to_session_and_timestamp[row.item_id].append((row.user_id, row.timestamp))

        item_interaction_counts = {item: len(sessions) for item, sessions in item_to_session_and_timestamp.items()}
        df_train["item_popularity"] = df_train["item_id"].map(item_interaction_counts)
        
        df_sessions = (
            df_train
            .groupby("user_id")["item_popularity"]
            .mean()
            .reset_index(name="avg_popularity")
        ).sort_values(by="avg_popularity", ascending=(args.method == "unpopular"))
        df_train = df_train.drop(columns="item_popularity")

    selected_idxs = []
    sessions_visited = 0

    for unlearning_fraction in sorted(args.unlearning_fractions):
        unlearn_count = int(n_interactions * unlearning_fraction)

        if args.method == "random":
            unlearn_rows = df_train.sample(n=unlearn_count, random_state=args.seed)
        else:
            selected_idxs = []
            row_count = 0
            
            for session in df_sessions["user_id"].iloc[sessions_visited:]:
                if row_count >= unlearn_count:
                    break
                sessions_visited += 1
                idx = session_indices.get(session, [])
                if len(idx) == 0:
                    continue

                selected_idxs.extend(idx)
                row_count += len(idx)
            
            unlearn_rows = df_train.iloc[selected_idxs].reset_index(drop=True)

        unlearn_rows = unlearn_rows.sort_values(by=["user_id", "timestamp"])
        unlearn_rows = unlearn_rows.rename(columns={
            "user_id": "user_id:token", 
            "item_id": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float"
        })

        output_name = f"goodreads_unlearn_pairs_{args.method}_seed_{args.seed}_unlearning_fraction_{unlearning_fraction}.inter"
        unlearn_rows.to_csv(output_name, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main()