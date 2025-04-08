import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class ItemBasedCF(GeneralRecommender):
    r"""
    Attempt to filter out valid/test interactions
    from `dataset.inter_feat` by "guessing" that the first portion
    of rows are train. Works for goodreads at least
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(ItemBasedCF, self).__init__(config, dataset)

        # A dummy param so the trainer won't crash with "empty param list".
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.num_users = dataset.num(dataset.uid_field)
        self.num_items = dataset.num(dataset.iid_field)

        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field

        # If there's a label field
        self.LABEL = None
        if dataset.label_field in dataset.inter_feat:
            self.LABEL = dataset.label_field

        # Hard-coded: read from config or default to [0.8, 0.1, 0.1]
        #  e.g. user might define a ratio for train/val/test
        if "split_ratio" in config:
            split_ratio = config["split_ratio"]  # e.g. [0.8, 0.1, 0.1]
        else:
            # fallback
            split_ratio = [0.8, 0.1, 0.1]

        train_ratio = split_ratio[0]
        inter_feat = dataset.inter_feat

        # total # of interactions
        total_inter = len(inter_feat)
        # approximate # of "train" interactions
        train_count = int(total_inter * train_ratio)

        # sort "inter_feat" if you assume 'TO' (time order) or that
        # the dataset is already sorted in some consistent way.
        # If you're not sure, or if 'RS' random splitting is used,
        # this won't match what RecBole actually does.
        #
        # For example, if 'TO' is used, RecBole typically sorts by time,
        # so you can do:
        # inter_feat.sort(by='timestamp') # If your data has a timestamp field
        # 
        # We'll skip that here, since we can't know how you want to sort.

        # We'll just take the first portion as "train"
        # We do this by slicing up to train_count
        train_rows = inter_feat[:train_count]

        user_indices = train_rows[self.USER_ID].numpy()
        item_indices = train_rows[self.ITEM_ID].numpy()

        if self.LABEL and self.LABEL in train_rows:
            ratings = train_rows[self.LABEL].numpy()
        else:
            ratings = [1.0] * len(user_indices)

        # Build user_ratings, item_ratings from *only these train_rows*
        self.user_ratings = {u: [] for u in range(self.num_users)}
        self.item_ratings = {i: [] for i in range(self.num_items)}

        for u, i, r in zip(user_indices, item_indices, ratings):
            self.user_ratings[u].append((i, float(r)))
            self.item_ratings[i].append((u, float(r)))

        # Precompute user-average rating from "train" portion
        user_sum = [0.0] * self.num_users
        user_count = [0] * self.num_users
        for u, r in zip(user_indices, ratings):
            user_sum[u] += r
            user_count[u] += 1

        avg_list = []
        for u in range(self.num_users):
            if user_count[u] > 0:
                avg_list.append(user_sum[u] / user_count[u])
            else:
                avg_list.append(0.0)

        self.register_buffer('user_avg', torch.tensor(avg_list, dtype=torch.float))

        # top_k
        if "top_k" in config:
            self.top_k = config["top_k"]
        else:
            self.top_k = 50

    def calculate_loss(self, interaction):
        # memory-based => no real trainable parameters => return zero
        return torch.zeros(1, device=self.device, requires_grad=True)

    def forward(self, user, item):
        return self._predict_internal(user, item)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self._predict_internal(user, item)

    def _predict_internal(self, batch_users, batch_items):
        preds = []
        for u, t_item in zip(batch_users.tolist(), batch_items.tolist()):
            cands = self.user_ratings.get(u, [])
            if not cands:
                preds.append(self.user_avg[u].item())
                continue

            sim_list = []
            rating_list = []
            for cand_item, rating in cands:
                sim = self._compute_similarity(t_item, cand_item)
                sim_list.append(sim)
                rating_list.append(rating)

            import torch
            sim_tensor = torch.tensor(sim_list, dtype=torch.float, device=self.device)
            rating_tensor = torch.tensor(rating_list, dtype=torch.float, device=self.device)

            if sim_tensor.numel() > self.top_k:
                topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                sim_tensor = topk_vals
                rating_tensor = rating_tensor[topk_idx]

            numerator = (sim_tensor * rating_tensor).sum()
            denom = torch.abs(sim_tensor).sum() + 1e-8
            if denom > 0:
                pred = numerator / denom
            else:
                pred = self.user_avg[u]
            preds.append(pred.item())

        return torch.tensor(preds, dtype=torch.float, device=self.device)

    def _compute_similarity(self, target_item, cand_item):
        r_list = self.item_ratings.get(target_item, [])
        c_list = self.item_ratings.get(cand_item, [])
        c_dict = {u: r for (u, r) in c_list}
        common = []
        for (u, r_t) in r_list:
            if u in c_dict:
                r_c = c_dict[u]
                common.append((u, r_t, r_c))
        if len(common) == 0:
            return 0.0

        numerator = 0.0
        sst = 0.0
        ssc = 0.0
        for (u, r_t, r_c) in common:
            avg = self.user_avg[u].item()
            dt = r_t - avg
            dc = r_c - avg
            numerator += dt * dc
            sst += dt * dt
            ssc += dc * dc

        denom = (sst**0.5)*(ssc**0.5) + 1e-8
        return float(numerator / denom)

    def full_sort_predict(self, interaction):
        # standard full ranking
        user = interaction[self.USER_ID]
        if user.numel() == 1:
            u_id = user.item()
            scores = []
            for t_item in range(self.num_items):
                cands = self.user_ratings.get(u_id, [])
                if not cands:
                    scores.append(self.user_avg[u_id].item())
                    continue

                sim_list = []
                rating_list = []
                for ci, rci in cands:
                    sim = self._compute_similarity(t_item, ci)
                    sim_list.append(sim)
                    rating_list.append(rci)

                import torch
                sim_tensor = torch.tensor(sim_list, device=self.device)
                rating_tensor = torch.tensor(rating_list, device=self.device)

                if sim_tensor.numel() > self.top_k:
                    topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                    sim_tensor = topk_vals
                    rating_tensor = rating_tensor[topk_idx]

                numerator = (sim_tensor * rating_tensor).sum()
                denom = sim_tensor.abs().sum() + 1e-8
                if denom > 0:
                    val = numerator / denom
                else:
                    val = self.user_avg[u_id]
                scores.append(val.item())

            return torch.tensor(scores, device=self.device).view(1, -1)
        else:
            all_scores = []
            for u_id in user.tolist():
                row_scores = []
                cands = self.user_ratings.get(u_id, [])
                for t_item in range(self.num_items):
                    if not cands:
                        row_scores.append(self.user_avg[u_id].item())
                        continue
                    sim_list = []
                    rating_list = []
                    for ci, rci in cands:
                        sim = self._compute_similarity(t_item, ci)
                        sim_list.append(sim)
                        rating_list.append(rci)
                    sim_tensor = torch.tensor(sim_list, device=self.device)
                    rating_tensor = torch.tensor(rating_list, device=self.device)

                    if sim_tensor.numel() > self.top_k:
                        topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                        sim_tensor = topk_vals
                        rating_tensor = rating_tensor[topk_idx]

                    numerator = (sim_tensor * rating_tensor).sum()
                    denom = sim_tensor.abs().sum() + 1e-8
                    if denom > 0:
                        val = numerator / denom
                    else:
                        val = self.user_avg[u_id]
                    row_scores.append(val.item())
                all_scores.append(row_scores)
            return torch.tensor(all_scores, device=self.device)


class ItemBasedCFOld(GeneralRecommender):
    r"""
    A memory-based Item-based CF implemented for RecBole.

    It does not learn embeddings through gradient descent;
    rather, it uses user-item interactions from the dataset to compute
    a similarity measure among items on the fly, then predicts a score
    using the top-K similar items the user has rated.
    """

    input_type = InputType.POINTWISE  # or PAIRWISE, depending on your usage

    def __init__(self, config, full_dataset, train_data):
        """
        Args:
            config: RecBole Config object
            full_dataset: The original dataset returned by create_dataset(config)
                (has the correct num_users, num_items)
            train_data: The training subset (DataLoader or Dataset) returned by data_preparation(config, dataset)[0]
        """
        # Pass the full dataset to super().__init__ so it sets up fields correctly
        super(ItemBasedCFOld, self).__init__(config, full_dataset)

        # TODO: remove this fix which adds a dummy parameter such that the Trainer does not throw an error when building a optimizer from an empty list
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        # 1) Get the real total number of users/items
        self.num_users = full_dataset.num(full_dataset.uid_field)
        self.num_items = full_dataset.num(full_dataset.iid_field)

        # 2) We'll read only the training interactions from train_data.dataset
        #    to build item-based CF
        splitted_dataset = train_data.dataset  # This is the splitted train subset
        inter_feat = splitted_dataset.inter_feat

        # RecBole field names
        self.USER_ID = splitted_dataset.uid_field
        self.ITEM_ID = splitted_dataset.iid_field
        self.LABEL = splitted_dataset.label_field  # might be None if no explicit rating

        # 3) Build user/item dictionaries from TRAIN ONLY
        user_indices = inter_feat[self.USER_ID].numpy()
        item_indices = inter_feat[self.ITEM_ID].numpy()
        if self.LABEL and self.LABEL in inter_feat:
            ratings = inter_feat[self.LABEL].numpy()
        else:
            ratings = [1.0] * len(user_indices)

        # Now fill user_ratings, item_ratings for *training users/items only*
        self.user_ratings = {u: [] for u in range(self.num_users)}
        self.item_ratings = {i: [] for i in range(self.num_items)}

        for u, i, r in zip(user_indices, item_indices, ratings):
            self.user_ratings[u].append((i, float(r)))
            self.item_ratings[i].append((u, float(r)))

        # 4) Compute user-average rating from training interactions
        user_sum = [0.0] * self.num_users
        user_count = [0] * self.num_users
        for u, r in zip(user_indices, ratings):
            user_sum[u] += r
            user_count[u] += 1

        avg_list = []
        for u in range(self.num_users):
            avg_list.append(user_sum[u] / user_count[u] if user_count[u] > 0 else 0.0)

        self.register_buffer('user_avg', torch.tensor(avg_list, dtype=torch.float))

        # Additional config (top_k, etc.)
        self.top_k = config['top_k']

    def compute_similarity(self, target_item: int, candidate_item: int) -> float:
        """
        Compute the adjusted cosine similarity between target_item and candidate_item on the fly.
        For all users who rated both items, subtract the user's average rating before computing.
        """
        # Ratings for the two items
        ratings_target = self.item_ratings.get(target_item, [])
        ratings_candidate = self.item_ratings.get(candidate_item, [])
        candidate_dict = {u: r for (u, r) in ratings_candidate}

        # Find common users
        common = []
        for u, r_t in ratings_target:
            if u in candidate_dict:
                r_c = candidate_dict[u]
                common.append((u, r_t, r_c))

        if len(common) == 0:
            return 0.0

        numerator = 0.0
        sum_sq_t = 0.0
        sum_sq_c = 0.0

        for (u, r_t, r_c) in common:
            avg = self.user_avg[u].item()
            diff_t = r_t - avg
            diff_c = r_c - avg
            numerator += diff_t * diff_c
            sum_sq_t += diff_t * diff_t
            sum_sq_c += diff_c * diff_c

        denominator = (sum_sq_t ** 0.5) * (sum_sq_c ** 0.5) + 1e-8
        return float(numerator / denominator)

    def forward(self, user: torch.Tensor, item: torch.Tensor):
        r"""
        forward() in RecBole is typically used to compute embeddings.
        Here, it can simply call the "predict" logic for convenience.

        Args:
            user (torch.Tensor): shape [batch_size]
            item (torch.Tensor): shape [batch_size]

        Returns:
            scores (torch.Tensor): predicted scores for each (user, item) pair in the batch
        """
        return self._predict_internal(user, item)

    def calculate_loss(self, interaction):
        r"""
        Since item-based CF is memory-based (no learnable parameters),
        there's no gradient-based training objective here. We simply return a
        zero loss to satisfy RecBole's training loop interface.
        """
        return torch.zeros(1, requires_grad=True, device=interaction[self.USER_ID].device)

    def predict(self, interaction):
        r"""
        This is the pointwise (or test-time) prediction interface required by RecBole.
        It returns predicted scores for the given (user, item) pairs.

        Args:
            interaction (Interaction): The interaction mini-batch, with fields:
                self.USER_ID and self.ITEM_ID.

        Returns:
            torch.Tensor: shape [batch_size], predicted scores
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self._predict_internal(user, item)

    def _predict_internal(self, batch_users: torch.Tensor, batch_items: torch.Tensor):
        """
        Re-usable internal function to compute item-based CF predictions for
        a batch of (user, item) pairs.
        """
        # We'll loop over the batch and compute the predictions item by item.
        preds = []
        user_list = batch_users.tolist()
        item_list = batch_items.tolist()

        for u, target_item in zip(user_list, item_list):
            # Retrieve all items rated by user u
            candidates = self.user_ratings.get(u, [])
            if not candidates:
                # If user has no interactions, fall back to user average
                preds.append(self.user_avg[u].item())
                continue

            sim_list = []
            rating_list = []

            # Compute similarity with each candidate item
            for cand_item, rating in candidates:
                sim = self.compute_similarity(target_item, cand_item)
                sim_list.append(sim)
                rating_list.append(rating)

            sim_tensor = torch.tensor(sim_list, dtype=torch.float, device=batch_users.device)
            rating_tensor = torch.tensor(rating_list, dtype=torch.float, device=batch_users.device)

            # Optionally keep only top_k neighbors
            if sim_tensor.numel() > self.top_k:
                topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                sim_tensor = topk_vals
                rating_tensor = rating_tensor[topk_idx]

            numerator = (sim_tensor * rating_tensor).sum()
            denominator = torch.abs(sim_tensor).sum() + 1e-8
            if denominator > 0:
                pred = numerator / denominator
            else:
                pred = self.user_avg[u]
            preds.append(pred.item())

        return torch.tensor(preds, device=batch_users.device, dtype=torch.float)

    def full_sort_predict(self, interaction):
        r"""
        Used in the full-sort setting. For each user in the batch, produce
        predictions for *all* items.

        This is typically used for ranking all items at test time.
        """
        user = interaction[self.USER_ID]
        # user might be of shape [batch_size], we assume we only do
        # one user at a time or unify them carefully.

        # If there's only one user in the batch:
        if user.numel() == 1:
            u = user.item()
            # We'll predict a score for *every* item
            scores = []
            for target_item in range(self.num_items):
                # Retrieve user's rated items
                candidates = self.user_ratings.get(u, [])
                if not candidates:
                    # fallback to user average
                    scores.append(self.user_avg[u].item())
                    continue

                sim_list = []
                rating_list = []
                for cand_item, rating in candidates:
                    sim = self.compute_similarity(target_item, cand_item)
                    sim_list.append(sim)
                    rating_list.append(rating)

                sim_tensor = torch.tensor(sim_list, dtype=torch.float, device=user.device)
                rating_tensor = torch.tensor(rating_list, dtype=torch.float, device=user.device)

                # top_k
                if sim_tensor.numel() > self.top_k:
                    topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                    sim_tensor = topk_vals
                    rating_tensor = rating_tensor[topk_idx]

                numerator = (sim_tensor * rating_tensor).sum()
                denominator = torch.abs(sim_tensor).sum() + 1e-8
                if denominator > 0:
                    score = numerator / denominator
                else:
                    score = self.user_avg[u]
                scores.append(score.item())

            return torch.tensor(scores, device=user.device, dtype=torch.float).view(1, -1)

        else:
            # If you have a batch of multiple users, you'd do similarly for each user.
            # For simplicity, here we show a loop. You can optimize if you like.
            all_scores = []
            user_list = user.tolist()
            for u in user_list:
                scores = []
                for target_item in range(self.num_items):
                    candidates = self.user_ratings.get(u, [])
                    if not candidates:
                        scores.append(self.user_avg[u].item())
                        continue

                    sim_list = []
                    rating_list = []
                    for cand_item, rating in candidates:
                        sim = self.compute_similarity(target_item, cand_item)
                        sim_list.append(sim)
                        rating_list.append(rating)

                    sim_tensor = torch.tensor(sim_list, dtype=torch.float, device=user.device)
                    rating_tensor = torch.tensor(rating_list, dtype=torch.float, device=user.device)
                    # top_k
                    if sim_tensor.numel() > self.top_k:
                        topk_vals, topk_idx = torch.topk(sim_tensor, self.top_k)
                        sim_tensor = topk_vals
                        rating_tensor = rating_tensor[topk_idx]

                    numerator = (sim_tensor * rating_tensor).sum()
                    denominator = torch.abs(sim_tensor).sum() + 1e-8
                    if denominator > 0:
                        score = numerator / denominator
                    else:
                        score = self.user_avg[u]
                    scores.append(score.item())

                all_scores.append(scores)

            return torch.tensor(all_scores, device=user.device, dtype=torch.float)
