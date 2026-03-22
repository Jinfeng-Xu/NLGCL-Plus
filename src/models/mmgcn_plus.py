# coding: utf-8
"""
MMGCN+NLGCL: Multi-modal Graph Convolution Network with Neighbor-aware Graph Contrastive Learning
In ACM MM`19 + NLGCL extension
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


class MMGCN_Plus(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCN_Plus, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']
        
        # NLGCL parameters
        self.cl_temp = config['cl_temp']
        self.cl_reg = config['cl_reg']
        self.alpha = config['alpha']
        self.n_layers = num_layer
        
        # dim_latent for modality feature projection
        self.dim_latent = dim_x

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=self.dim_latent, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=self.dim_latent, device=self.device)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal

        self.result = representation
        return representation

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
        for layer_idx in range(1, self.n_layers + 1):
            cur_embedding_u, cur_embedding_i = torch.split(embeddings_list[layer_idx], [self.n_users, self.n_items])
            
            cl_u = cl_u + self.InfoNCE(cur_embedding_i[pos_item], ego_embedding_u[user],
                                     ego_embedding_u[user]) + 1e-6

            cl_i = cl_i + self.InfoNCE(cur_embedding_u[user], ego_embedding_i[pos_item],
                                     ego_embedding_i[pos_item]) + 1e-6
            # update embeddings
            ego_embedding_u, ego_embedding_i = cur_embedding_u, cur_embedding_i
        return cl_u, cl_i

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        # Get multi-layer embeddings for contrastive learning
        representation, embeddings_list = self.forward_with_layers()
        
        user_score = representation[user_tensor]
        item_score = representation[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        
        # BPR loss
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        
        # NLGCL contrastive loss
        ego_cl_loss_u, ego_cl_loss_i = self.neighbor_cl_loss(embeddings_list, batch_users, pos_items - self.n_users, neg_items - self.n_users)
        ego_cl_loss = self.alpha * ego_cl_loss_u + (1 - self.alpha) * ego_cl_loss_i
        cl_loss = ego_cl_loss * self.cl_reg
        
        return bpr_loss + reg_loss + cl_loss

    def forward_with_layers(self):
        """Forward pass that returns multi-layer embeddings for contrastive learning"""
        all_embeddings = self.id_embedding
        embeddings_list = [all_embeddings]
        
        # Get modality features
        modality_features = None
        if self.v_feat is not None:
            modality_features = self.v_gcn(self.v_feat, self.id_embedding, return_layers=True)
        if self.t_feat is not None:
            t_features = self.t_gcn(self.t_feat, self.id_embedding, return_layers=True)
            if modality_features is None:
                modality_features = t_features
            else:
                # Average modality features
                modality_features = [(v + t) / 2 for v, t in zip(modality_features, t_features)]
        
        if modality_features is not None:
            all_embeddings = modality_features[-1]
            embeddings_list = modality_features
        
        result = all_embeddings
        self.result = result
        return result, embeddings_list

    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding, return_layers=False):
        if return_layers:
            return self.forward_with_layers(features, id_embedding)
        
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x

    def forward_with_layers(self, features, id_embedding):
        """Forward pass that returns embeddings at each layer for contrastive learning"""
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)
        
        embeddings_list = [x]

        # Layer 1
        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)
        embeddings_list.append(x)

        # Layer 2
        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)
        embeddings_list.append(x)

        # Layer 3
        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)
        embeddings_list.append(x)

        return embeddings_list


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
