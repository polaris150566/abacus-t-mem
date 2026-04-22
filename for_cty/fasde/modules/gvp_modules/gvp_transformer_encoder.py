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
from .features import GVPInputFeaturizer, DihedralFeatures
from .gvp_encoder import GVPEncoder
from .transformer_layer import TransformerEncoderLayer
from .util import nan_to_num, get_rotation_frames, rotate, rbf
from torch.utils.checkpoint import checkpoint

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import GVPGraphEmbedding
from ._modules import ConvLayer
from .gvp_utils import unflatten_graph


class GraphDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        conv_activations = (F.relu, torch.sigmoid)
        
        self.encoder_layers = nn.ModuleList(
                ConvLayer(
                    2*args.gvp_node_hidden_dim_scalar,
                    args.gvp_edge_hidden_dim_scalar,
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-6,
                    layernorm=True,
                ) 
            for i in range(3)
        )

    def forward(self, node_embeddings, edge_index, edge_embeddings):
        bs, _, dims = node_embeddings.shape
        node_embeddings = node_embeddings.reshape(-1, dims)
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)
        node_embeddings = node_embeddings.reshape(bs, -1, dims)
        return node_embeddings


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
        self.embed_gvp_input_features = nn.Linear(24, embed_dim)
        # self.embed_gvp_input_features = nn.Linear(6, embed_dim)

        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        gvp_args = argparse.Namespace()
        for k, v in vars(args).items():
            if k.startswith("gvp_"):
                setattr(gvp_args, k[4:], v)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)
        # gvp_out_dim = gvp_args.node_hidden_dim_scalar 
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

    
        self.layer_norm = nn.LayerNorm(embed_dim)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self,batch, coords, padding_mask, confidence, positions):
        """
        Args:
            coords: N, CA, C backbone coordinates in shape length x 3 (atoms) x 3 
            padding_mask: boolean Tensor (true for padding) of shape length
            confidence: confidence scores between 0 and 1 of shape length
        """
        components = dict()
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)
        mask_tokens = (
            padding_mask * self.dictionary.padding_idx + 
            ~padding_mask * self.dictionary.get_idx("<mask>")
        )
        components["tokens"] = self.embed_tokens(mask_tokens) * self.embed_scale
        components["diherals"] = self.embed_dihedrals(coords)
        # GVP encoder
        (gvp_out_scalars, gvp_out_vectors), edge_embeddings, edge_index = self.gvp_encoder(batch, coords,
                coord_mask, padding_mask, confidence)
        R = get_rotation_frames(coords)
        # Rotate to local rotation frame for rotation-invariance

        lig_mask = batch['lig_mask']
        R = R*(1-lig_mask[...,None,None])

        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)
        components["confidence"] = self.embed_confidence(
             rbf(confidence, 0., 1.))
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(batch,
            coords, coord_mask, with_coord_mask=False ,all_node=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_input_features"] = self.embed_gvp_input_features(features)
        embed = sum(components.values())
        x = embed
        # x = x + self.embed_positions(positions)
        x = self.dropout_module(x)
        return x, components, edge_embeddings, edge_index

    def forward(
        self,
        batch,
        coords,
        encoder_padding_mask,
        confidence,
        positions,
        return_all_hiddens: bool = False,
        use_gradient_checkpoint: bool = False,
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
        # if use_gradient_checkpoint and self.training:
        #     x, encoder_embedding = checkpoint(
        #         self.forward_embedding,
        #         coords, encoder_padding_mask, confidence,positions
        #     )
        # else:
        x, encoder_embedding, edge_embeddings, edge_index = self.forward_embedding(batch, coords,
                encoder_padding_mask, confidence, positions)
        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        max_len = batch['seg_len'][:,0].max()
        lig_mask = batch['lig_mask'].bool()
        rec_mask = batch['node_mask'] * (1 - lig_mask.float())
        # rec_x = (x*rec_mask[...,None])[:,:max_len]
        # encoder_padding_mask= (encoder_padding_mask |(~rec_mask.bool()))[:, :max_len]
        # x = rec_x
        
        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
            "edge_embeddings": edge_embeddings,
            'edge_index': edge_index,
        }
