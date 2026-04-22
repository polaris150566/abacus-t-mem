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

from ..modules import SinusoidalPositionalEmbeddingv1, LearnedPositionalEmbedding
from .features import GVPInputFeaturizer, DihedralFeatures
from .gvp_encoder import GVPEncoder
from .transformer_layer import TransformerEncoderLayer
from .util import nan_to_num, get_rotation_frames, rotate, rbf


class GVPTransformerScoreNet(nn.Module):
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
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        gvp_args = argparse.Namespace()
        for k, v in vars(args).items():
            if k.startswith("gvp_"):
                setattr(gvp_args, k[4:], v)
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # diffussion input and outputs
        self.embed_t = nn.Embedding(
            args.diff_T + 1, args.diff_t_embedding_dim
        )

        self.input_proj = nn.Linear(
            embed_dim + args.diff_t_embedding_dim + args.output_dim,
            embed_dim
        )

        self.regression_head = nn.Linear(
            embed_dim, args.output_dim
        )

        classification_dim = embed_dim if args.classification_hidden else args.output_dim
        self.classification_head = nn.Linear(classification_dim, args.token_vocab_size)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward_embedding(self, coords, padding_mask, confidence, positions):
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
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, padding_mask, confidence)
        R = get_rotation_frames(coords)
        # Rotate to local rotation frame for rotation-invariance
        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)

        components["confidence"] = self.embed_confidence(
             rbf(confidence, 0., 1.))

        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())
        # for k, v in components.items():
        #     print(k, torch.mean(v, dim=(0,1)), torch.std(v, dim=(0,1)))

        x = embed
        # x = x + self.embed_positions(mask_tokens)
        x = x + self.embed_positions(positions)

        x = self.dropout_module(x)
        return x, components 

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        positions,
        t,
        xt,
        gvp_encoder_output = None,
        return_all_hiddens: bool = False,
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
        if gvp_encoder_output is None:
            x, encoder_embedding = self.forward_embedding(coords,
                    padding_mask, confidence, positions)
        else:
            x, encoder_embedding = gvp_encoder_output
        gvp_output = x.clone()

        # diffusion input
        t_embed = self.embed_t(t)
        t_embed = t_embed[:, None].tile(1, x.shape[1], 1)
        x = torch.cat(
            [x, xt, t_embed], dim=-1
        )
        x = self.input_proj(x)

        # account for padding while computing the representation
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x0 = self.regression_head(x)

        if self.args.classification_hidden:
            logits = self.classification_head(x)
        else:
            logits = self.classification_head(x0)

        return {
            "x0": x0,
            "logits": logits,
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
            "gvp_output": gvp_output,
        }
