# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import GVPGraphEmbedding
from .gvp_modules import GVPConvLayer, LayerNorm, GVPConvCrossLayer
from .gvp_utils import unflatten_graph
from torch.utils.checkpoint import checkpoint



            
class GVPEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                args.edge_hidden_dim_vector)
        conv_activations = (F.relu, torch.sigmoid)

        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_hidden_dim, edge_hidden_dim, drop_rate=args.dropout, vector_gate=True, attention_heads=0, n_message=3, \
            conv_activations=conv_activations,n_edge_gvps=0,eps=1e-8,layernorm=True,)
            for i in range(args.num_encoder_layers)
        )

    def forward(self,batch, use_gradient_checkpoint = False):
        coords = batch['atom_positions_t']
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
               batch)
        for layer in self.encoder_layers:
            node_embeddings, edge_embeddings = layer(node_embeddings,
                        edge_index, edge_embeddings)
       
        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
       
        return node_embeddings

