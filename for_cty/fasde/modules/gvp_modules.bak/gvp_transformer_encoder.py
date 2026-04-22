# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..modules import SinusoidalPositionalEmbeddingv1
from .features import GVPInputFeaturizer
from .gvp_encoder import GVPEncoder
from .gvp_transformer_layer import GVPTransformerLayer,GVPTransoformerStackLayer
from .util import nan_to_num, get_rotation_frames, rotate, rbf
from torch.utils.checkpoint import checkpoint
from .gvp_modules import GVP,LayerNorm




class GVPTransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()
        self.args = args
        self.dictionary = dictionary

        self.dropout_module = nn.Dropout(args.dropout)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbeddingv1(
            embed_dim,
            self.padding_idx,
        )
        self.embed_chain = nn.Embedding(10,embed_dim)

        gvp_args = argparse.Namespace()
        for k, v in vars(args).items():
            if k.startswith("gvp_"):
                setattr(gvp_args, k[4:], v)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar 
        
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.node_in=[embed_dim, gvp_args.node_hidden_dim_vector]
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm_out = LayerNorm((embed_dim,embed_dim))
        self.layer_norm = LayerNorm((embed_dim,embed_dim))


    def build_encoder_layer(self, args):
        return GVPTransoformerStackLayer(args,node_in=[256,256])

    def forward_embedding(self, batch):
        """
        Args:
            coords: N, CA, C backbone coordinates in shape length x 3 (atoms) x 3 
            padding_mask: boolean Tensor (true for padding) of shape length
            confidence: confidence scores between 0 and 1 of shape length
        """
        gvp_out_scalar, gvp_out_vector  = self.gvp_encoder(batch)
        gvp_out_scalar = self.embed_gvp_output(gvp_out_scalar) + self.embed_positions(batch['residx']) + self.embed_chain(batch['chainidx'])
        gvp_out_scalar = self.dropout_module(gvp_out_scalar)
        node  = (gvp_out_scalar,  gvp_out_vector)
        node =  self.layer_norm_out(node)
        return node
 
    def forward(
        self,
        batch,
        return_all_hiddens: bool = False,
        use_gradient_checkpoint: bool = True,
    ):
        """
        Args:
            coords (Tensor): backbone coordinates
                shape batch_size x num_residues x num_atoms (3 for N, CA, C) x 3
            encoder_padding_mask (ByteTensor): the positions of
                  padding elements of shape `(batch_size x num_residues)`
            confidence (Tensor): the confidence score of shape (batch_size x
                num_residues). The value is between 0. and 1. for each residue
                coordinate, or -1. if no coordinate is given
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(num_residues, batch_size, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch_size, num_residues)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch_size, num_residues, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(num_residues, batch_size, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
       
        batch['padding_mask'] = (batch['node_mask'] == 0.0)
        batch['lig_padding_mask'] = (batch['lig_node_mask'] == 0.0)
        batch['rec_padding_mask'] = (batch['rec_node_mask'] == 0.0)
        node = self.forward_embedding(batch)
        encoder_states = []
        use_gradient_checkpoint = False
        # encoder layers
        for i, layer in enumerate(self.layers):
            if use_gradient_checkpoint and self.training:
                node = checkpoint(
                    layer,node, batch
                )
            else:
                node = layer(
                    node, batch,index=i
                )
        node = self.layer_norm(node)

        return {
            "encoder_out": node,  # T x B x C
            "encoder_states": encoder_states,  # List[T x B x C]
        }
