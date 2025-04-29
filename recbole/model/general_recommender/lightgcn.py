# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import from_scipy_sparse_matrix

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # 1) hyperparams
        self.latent_dim = config["embedding_size"]
        self.n_layers   = config["n_layers"]
        self.reg_weight = config["reg_weight"]

        # 2) embedding layers (register as parameters)
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)

        # 3) instantiate RecBole losses
        self.mf_loss  = BPRLoss()
        self.reg_loss = EmbLoss()

        # 4) build PyG graph once
        R = dataset.inter_matrix(form="coo").astype(np.float32)
        # bipartite adjacency: top-right R, bottom-left Rᵀ
        mat = sp.bmat([[None,          R         ],
                       [R.transpose(), None      ]], format="coo")
        edge_index, edge_weight = from_scipy_sparse_matrix(mat)

        # move to device
        self.edge_index  = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)

        # 5) PyG LightGCN layers (LGConv == LightGCN operator)
        self.convs = torch.nn.ModuleList([
            LGConv(normalize=False) for _ in range(self.n_layers)
        ])

        # 6) parameter init
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        # concatenate user+item embeddings
        x = torch.cat([self.user_embedding.weight,
                       self.item_embedding.weight], dim=0)          # [N, D]
        all_embeddings = [x]

        # message passing via LGConv
        for conv in self.convs:
            x = conv(x, self.edge_index, self.edge_weight)             # PyG does the spmm
            all_embeddings.append(x)

        # layer‐wise mean
        h = torch.stack(all_embeddings, dim=1).mean(dim=1)             # [N, D]
        user_emb, item_emb = torch.split(h, [self.n_users, self.n_items], dim=0)
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb = self.forward()
        u_e  = user_emb[user]
        pos_e = item_emb[pos_item]
        neg_e = item_emb[neg_item]

        # BPR and regularization
        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        mf = self.mf_loss(pos_scores, neg_scores)

        # reg on the “ego” embeddings
        u0  = self.user_embedding(user)
        p0  = self.item_embedding(pos_item)
        n0  = self.item_embedding(neg_item)
        reg = self.reg_loss(u0, p0, n0)

        return mf + self.reg_weight * reg

    def predict(self, interaction):
        u_idx = interaction[self.USER_ID]
        i_idx = interaction[self.ITEM_ID]
        user_emb, item_emb = self.forward()
        return (user_emb[u_idx] * item_emb[i_idx]).sum(dim=1)

    def full_sort_predict(self, interaction):
        u_idx = interaction[self.USER_ID]
        if not hasattr(self, "restore_user_e") or not hasattr(self, "restore_item_e") or self.restore_user_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        scores = torch.matmul(self.restore_user_e[u_idx], self.restore_item_e.t())
        return scores.view(-1)
