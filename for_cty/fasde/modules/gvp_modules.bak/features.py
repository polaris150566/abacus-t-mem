# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Portions of this file were adapted from the open source code for the following
# two papers:
#
#   Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative
#   models for graph-based protein design. Advances in Neural Information
#   Processing Systems, 32.
#
#   Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2020).
#   Learning from Protein Structure with Geometric Vector Perceptrons. In
#   International Conference on Learning Representations.
#
# MIT License
# 
# Copyright (c) 2020 Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael Townshend, Ron Dror
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ================================================================
# The below license applies to the portions of the code (parts of 
# src/datasets.py and src/models.py) adapted from Ingraham, et al.
# ================================================================
# 
# MIT License
# 
# Copyright (c) 2019 John Ingraham, Vikas Garg, Regina Barzilay, Tommi Jaakkola
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvp_utils import flatten_graph, flatten_graph_cross
from .gvp_modules import GVP, LayerNorm, GVPWithAAtype
from .util import normalize, norm, nan_to_num, rbf
from ..embedders import get_timestep_embedding

def length_to_mask(lens):
    max_len = lens.max()
    indices = torch.arange(max_len).expand(len(lens), max_len).to(lens.device)
    mask = indices < lens.unsqueeze(1)
    indices = indices*mask
    return indices, mask

class GVPInputFeaturizer(nn.Module):
    @staticmethod
    def get_node_features( batch):
        node_scalar_features=batch['single']
        lig_mask = batch['lig_mask'].bool()
        coords = batch['atom_positions_t']
        X_ca = coords[:, :, 1]
        rec_mask = batch['node_mask'] * (1 - lig_mask.float())
        orientations = GVPInputFeaturizer._orientations(X_ca)
        orientations = orientations * rec_mask[..., None, None]
        coords_to_ca = (coords - X_ca[..., None, :]) * batch['atom_mask'][..., None]
        node_vector_features = torch.cat([coords_to_ca, orientations], dim=-2)
        return node_scalar_features, node_vector_features


    @staticmethod
    def _orientations(X):
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
    


    @staticmethod
    def _positional_embeddings(edge_index, 
                               num_embeddings=None,
                               num_positional_embeddings=16,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _dist(X, coord_mask, padding_mask, top_k_neighbors, eps=1e-8):
        """ Pairwise euclidean distances """
        bsz, maxlen = X.size(0), X.size(1)
        coord_mask_2D = torch.unsqueeze(coord_mask,1) * torch.unsqueeze(coord_mask,2)
        residue_mask = ~padding_mask
        residue_mask_2D = torch.unsqueeze(residue_mask,1) * torch.unsqueeze(residue_mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = coord_mask_2D * norm(dX, dim=-1)
    
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D) + (~coord_mask_2D) * (1e8 + Dseq*1e6) + (
            ~residue_mask_2D) * (1e10)
        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            E_idx = seqpos.repeat(
                    *D_neighbors.shape[:-1], 1)
        else:
            # Identify k nearest neighbors (including self)
            k = min(top_k_neighbors, X.size(1))
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
    
        coord_mask_neighbors = (D_neighbors < 5e7)
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, coord_mask_neighbors, residue_mask_neighbors


class GVPGraphEmbedding(GVPInputFeaturizer):

    def __init__(self, args):
        super().__init__()
        self.top_k_neighbors = args.top_k_neighbors
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        node_input_dim = (255, 6)
        edge_input_dim = (156, 1)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
       
        self.embed_node = GVPWithAAtype(node_input_dim, node_hidden_dim, activations=(None, None))
        self.node_layer_norm = LayerNorm(node_hidden_dim, eps=1e-4)


        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-7)
        )

        self.t_embed_dim = 32
        self.n_rbf_bins = 100

    def forward(self,batch):
        with torch.no_grad():
            node_feat_s, node_feat_v = self.get_node_features(batch)
            (edge_feat_s, edge_feat_v), edge_index = self.get_edge_features(batch)
            t = (1000*batch['t']).long()
            t_embed = get_timestep_embedding(t,self.t_embed_dim)
        _, node_len, _ = node_feat_s.shape
        node_feat_s = torch.concat([node_feat_s, t_embed[:,None].tile(1, node_len, 1)],dim=-1)
        node_feat= (node_feat_s, node_feat_v)
      
        _, node_len_e, _ = edge_feat_s.shape
        edge_feat_s = torch.concat([edge_feat_s,t_embed[:,None].tile(1, node_len_e, 1)],dim=-1)
        edge_feat = (edge_feat_s, edge_feat_v)
       
        node_embeddings = self.embed_node(node_feat, batch['aatype'])
        node_embeddings = self.node_layer_norm(node_embeddings)
        edge_embeddings= self.embed_edge(edge_feat)
      
        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
    
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, batch):
        coords = batch['atom_positions_t']
        coord_mask = batch[f'atom_mask']
        padding_mask = batch[f'padding_mask']
        X_ca = coords[:, :, 1]
        # Get distances to the top k neighbors
        coord_mask = coord_mask[:, :, 1].bool()
        E_dist, E_idx, E_coord_mask, E_residue_mask = GVPInputFeaturizer._dist(
                X_ca, coord_mask, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package 
        # dest = E_idx
        src = E_idx
        B, L, k = E_idx.shape[:3]
        # src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        dest = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)

        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF 
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20., n_bins=self.n_rbf_bins)

        # Calculate relative orientation 
        X_dest = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_src = torch.gather(
            X_ca,
            1,
            edge_index[0, :, :].unsqueeze(-1).expand([B, L*k, 3])
        )
        coord_mask_dest = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        coord_mask_src = torch.gather(coord_mask, 1, edge_index[0, :, :].expand([B, L*k]))
        E_vectors = X_src - X_dest
        # For the ones without coordinates, substitute in the average vector
        E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1,
                keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
        E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~(E_coord_mask)
        # Normalize and remove nans 
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors).unsqueeze(-2)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present 
        edge_s = torch.cat([
            edge_s,
            (~coord_mask_src).float().unsqueeze(-1),
            (~coord_mask_dest).float().unsqueeze(-1),
        ], dim=-1)
        edge_index[:, ~E_residue_mask] = -1
        if self.remove_edges_without_coords:
            edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        #####
        # Covalent edges
        cov_edge_index = batch[f'edge_index'].transpose(-1,-2)
        cov_edge_attr = batch[f'edge_attr']
        cov_edge_dim = cov_edge_attr.shape[-1]
        cov_edge_mask = (cov_edge_index.sum(1) > 0).float()

        src, dest = cov_edge_index[:, 0], cov_edge_index[:, 1]
        X_src = torch.gather(X_ca, 1, src[..., None].tile(1, 1, 3))
        X_dest = torch.gather(X_ca, 1, dest[..., None].tile(1, 1, 3))
        cov_vec = X_src - X_dest
        cov_dist = norm(cov_vec, dim=-1)
        cov_D_rbf = rbf(cov_dist, 0., 20., n_bins=self.n_rbf_bins)
        cov_vec = normalize(cov_vec).unsqueeze(-2)
        cov_D_rbf = cov_D_rbf * cov_edge_mask[..., None]
        cov_vec = cov_vec * cov_edge_mask[..., None, None]
        cov_edge_attr = torch.cat([cov_edge_attr, cov_D_rbf], dim=-1)

        edge_index = edge_index.transpose(0, 1) 
        
        mask1 = (edge_index[:,0] - batch['seg_len'][:,0].unsqueeze(-1) < 0) & (edge_index[:,1] - batch['seg_len'][:,0].unsqueeze(-1) >=0)
        mask2 = (edge_index[:,0] - batch['seg_len'][:,0].unsqueeze(-1) >=0 ) & (edge_index[:,1] - batch['seg_len'][:,0].unsqueeze(-1) <0)
        mask=mask1 | mask2
        edge_s=torch.concat([edge_s,mask.float()[...,None]],dim=-1)

        num_knn_edges = edge_index.shape[2]
        num_cov_edges = cov_edge_index.shape[2]
        knn_edge_dim = edge_s.shape[-1]
        cov_full_edge_dim = cov_edge_attr.shape[-1]
        
        edge_attr = torch.zeros((edge_index.shape[0], num_knn_edges+num_cov_edges, knn_edge_dim + cov_edge_dim)).to(edge_s.device)
        edge_attr[:, :num_knn_edges, cov_edge_dim:] = edge_s
        edge_attr[:, num_knn_edges:, :cov_full_edge_dim] = cov_edge_attr
        edge_s = edge_attr

        cov_edge_index = (cov_edge_index * cov_edge_mask[:, None] - (1.0 - cov_edge_mask[:, None])).long()      
        edge_index = torch.cat([edge_index, cov_edge_index], dim=-1)
        edge_v = torch.cat([edge_v, cov_vec], dim=1)
      
        return (edge_s, edge_v), edge_index
