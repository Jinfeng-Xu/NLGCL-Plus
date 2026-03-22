# coding: utf-8
# 
"""
DualGNN+NLGCL: Dual Graph Neural Network with Neighbor-aware Graph Contrastive Learning
IEEE Transactions on Multimedia 2021 + NLGCL extension
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class DualGNN_Plus(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DualGNN_Plus, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        self.construction = 'weighted_sum'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        
        # NLGCL parameters
        self.cl_temp = config['cl_temp']
        self.cl_reg = config['cl_reg']
        self.alpha = config['alpha']
        self.n_layers = config['n_layers']

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']), allow_pickle=True).item()

        # packing interaction in training into edge_index
        edge_index = self.pack_edge_index(dataset.inter_matrix(form='coo').astype(np.float32))
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i.data, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 3, self.dim_latent)

        if self.v_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)).to(self.device)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)).to(self.device)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device, features=self.t_feat)

        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes = interaction[0]
        # 使用 clone 避免修改原始 interaction 数据
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users
        representation = None
        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            representation = self.v_rep
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            if representation is None:
                representation = self.t_rep
            else:
                representation += self.t_rep

        if self.construction == 'weighted_sum':
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)

        item_rep = representation[self.num_user:]
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep = user_rep + h_u1
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def InfoNCE(self, view1, view2, view):
        """InfoNCE contrastive loss"""
        view1, view2, view = F.normalize(view1), F.normalize(view2), F.normalize(view)
        pos_score = torch.mul(view1, view2).sum(dim=1)
        pos_score = torch.exp(pos_score / self.cl_temp)
        ttl_score = torch.matmul(view1, view.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.cl_temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score).sum()
        return cl_loss

    def neighbor_cl_loss(self, embeddings_list, user, pos_item, neg_item):
        """Neighbor-aware contrastive loss from NLGCL"""
        ego_embedding_u, ego_embedding_i = torch.split(embeddings_list[0], [self.n_users, self.n_items])
        cl_u = 0
        cl_i = 0
        for layer_idx in range(1, len(embeddings_list)):
            cur_embedding_u, cur_embedding_i = torch.split(embeddings_list[layer_idx], [self.n_users, self.n_items])
            
            cl_u = cl_u + self.InfoNCE(cur_embedding_i[pos_item], ego_embedding_u[user],
                                     ego_embedding_u[user]) + 1e-6

            cl_i = cl_i + self.InfoNCE(cur_embedding_u[user], ego_embedding_i[pos_item],
                                     ego_embedding_i[pos_item]) + 1e-6
            ego_embedding_u, ego_embedding_i = cur_embedding_u, cur_embedding_i
        return cl_u, cl_i

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        bpr_loss = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
        
        # NLGCL contrastive loss
        # Get multi-layer embeddings from GCN
        v_embeddings_list = self.v_gcn.get_embeddings_list(self.edge_index_dropv, self.edge_index, self.v_feat) if self.v_feat is not None else None
        t_embeddings_list = self.t_gcn.get_embeddings_list(self.edge_index_dropt, self.edge_index, self.t_feat) if self.t_feat is not None else None
        
        # Average modality embeddings
        if v_embeddings_list is not None and t_embeddings_list is not None:
            embeddings_list = [(v + t) / 2 for v, t in zip(v_embeddings_list, t_embeddings_list)]
        elif v_embeddings_list is not None:
            embeddings_list = v_embeddings_list
        else:
            embeddings_list = t_embeddings_list
        
        if embeddings_list is not None:
            ego_cl_loss_u, ego_cl_loss_i = self.neighbor_cl_loss(embeddings_list, user, interaction[1], interaction[2])
            ego_cl_loss = self.alpha * ego_cl_loss_u + (1 - self.alpha) * ego_cl_loss_i
            cl_loss = ego_cl_loss * self.cl_reg
            return bpr_loss + reg_loss + cl_loss
        
        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k
            user_graph_index.append(user_graph_sample)

        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)
            self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = h + x + h_1
        return x_hat, self.preference
    
    def get_embeddings_list(self, edge_index_drop, edge_index, features):
        """Get embeddings at each layer for contrastive learning"""
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        
        embeddings_list = [x]
        
        h = self.conv_embed_1(x, edge_index)
        embeddings_list.append(h)
        
        h_1 = self.conv_embed_1(h, edge_index)
        embeddings_list.append(h + x + h_1)
        
        return embeddings_list


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
