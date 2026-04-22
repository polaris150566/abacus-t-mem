# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from scipy.spatial import transform

from ..data import Alphabet

from .gvp_encoder import GVPEncoder
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_decoder import TransformerDecoder
from .gvp_modules import GVP,GVPWithAAtype
from .util import rotate, CoordBatchConverter 

def length_to_mask(lens,max_len=None):
    if max_len is None:
        max_len = lens.max()
    mask = torch.arange(max_len).expand(len(lens), max_len).to(lens.device)
    mask = mask < lens.unsqueeze(1)
    return mask.float()

class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        activations = (F.relu, torch.sigmoid)
        embed_dim = args.encoder_embed_dim
        # self.gvp_rec = GVPWithAAtype([embed_dim, embed_dim], [embed_dim, 37], activations=activations, init_zero=False,av_init=True)
        self.gvp = GVP([embed_dim, embed_dim], [embed_dim, 4], activations=activations, init_zero=False)


    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        batch,
        return_all_hiddens: bool = False,
        use_gradient_checkpoint: bool = True,
        fix_bb = True,
    ):  
        
        encoder_out = self.encoder(batch,
            return_all_hiddens=return_all_hiddens,use_gradient_checkpoint=use_gradient_checkpoint)
        decoder_input = encoder_out["encoder_out"]
        _, full_atom_output = self.gvp(decoder_input)
        output = full_atom_output * batch['atom_mask'][..., None]
        sigma_t = batch['sigma_t']
        x_t = batch['atom_positions_t']
        x_0_ = (output * sigma_t + x_t ) 
        x_0 = batch['atom_positions_0']
        # if fix_bb:
        #     bbmask = torch.zeros((37)).to(x_0_.device)
        #     bbmask[:3] = 1.
        #     bbmask[4] = 1.
        #     lens = batch['seg_len'][:,0]
        #     rec_mask = length_to_mask(lens, x_0.shape[1])
        #     bbmask = rec_mask[...,None] * bbmask[None,None,:]
        #     x_0_ = x_0_ * (1 - bbmask[..., None]) + x_0 * bbmask[..., None]
        batch['x_0_'] = x_0_
        
        return output