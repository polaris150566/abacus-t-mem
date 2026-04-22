# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..multihead_attention import MultiheadAttention
from .gvp_modules import GVP,LayerNorm,layernorm_vec, Dropout
from torch import Tensor
from .util import rbf


class GVPTransoformerStackLayer(nn.Module):
    def __init__(self,args,bias_weight=-1e-2,node_in=(1024,256)) -> None:
        super().__init__()

        self.gvp_transformer= GVPTransformerLayer(args,bias_weight,node_in)
       
    def forward(self, node, batch,index=None):
        batch['cent'] = batch['atom_positions_t'][...,1,:]
        node = self.gvp_transformer( node, encoder_padding_mask = batch['padding_mask'], coord_a = batch['cent'],)
        
        return node

class GVPTransformerLayer(nn.Module):
    """Encoder layer block.
    `layernorm -> dropout -> add residual`

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self,args,bias_weight=-0.05,node_in=(1024,256)):
        super().__init__()
        
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim,args)
        self.vec_attn = self.build_vec_attention(self.embed_dim,args)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.gvp_layer_norm = LayerNorm(node_in)

        self.dropout_module = nn.Dropout(args.dropout)
        self.dropout_node = Dropout(args.dropout)

        ff_func = []
        n_feedforward=2
        vector_gate = True
        node_dims = node_in
        if n_feedforward == 1:
            ff_func.append(GVP(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP(node_dims, hid_dims, vector_gate=vector_gate))
            for i in range(n_feedforward-2):
                ff_func.append(GVP(hid_dims, hid_dims, vector_gate=vector_gate))
            ff_func.append(GVP(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

        node_out = node_in
        activations = (F.relu, torch.sigmoid)
        self.gvp = GVP(node_in, node_out, activations=activations)
        self.vec_embed_s = GVP(node_out, (node_out[0], args.encoder_attention_heads), activations=activations)
        self.vec_embed_v = GVP(node_out, (node_out[0], args.encoder_attention_heads), activations=activations)
        self.bias_weight = bias_weight
    def build_self_attention(self, embed_dim,args ):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def build_vec_attention(self, embed_dim,args ):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
        )

    def residual_connection_tuple(self, x, residual):
        return ((residual[0] + x[0]), (residual[1] + x[1]))

    def forward(
        self,
        s_vec,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        coord_a=None,
    ):
        s, vec = s_vec
        residual = s.transpose(0,1)
        residual_vec = vec
        # s_vec = self.gvp(s_vec) 
        # s, vec = s_vec 
        _, vectors_head = self.vec_embed_s(s_vec)
        vectors_head = vectors_head.transpose(1,2) 
        vectors_head = vectors_head + coord_a.unsqueeze(1)    
        sub_v = vectors_head[...,None,:] - vectors_head[...,None,:,:]
        attn_bias_v = self.bias_weight * sub_v.norm(dim=-1)**2 
        s = s.transpose(0,1)
        s = self.self_attn_layer_norm(s)
        s, _ = self.self_attn(
            query=s, key=s, value=s,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_bias=attn_bias_v,
            attn_mask=attn_mask
        )
        s = self.dropout_module(s)
        s = s + residual
        s_vec = (s.transpose(0,1), vec)
        _, vectors_head = self.vec_embed_v(s_vec)
        vectors_head = vectors_head.transpose(1,2) 
        vectors_head = vectors_head + coord_a.unsqueeze(1)    
        sub_v = vectors_head[...,None,:] - vectors_head[...,None,:,:]
        attn_bias_v = self.bias_weight * sub_v.norm(dim=-1)**2 
        
        vec = vec.transpose(0,1)
        vec, _ = self.vec_attn(
            query=s, key=s, value=vec,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_bias=attn_bias_v,
            attn_mask=attn_mask
        )

        vec = vec.transpose(0,1)
        vec = vec + residual_vec
        s_vec = (s.transpose(0,1),vec)
        residual_sv = s_vec
        s_vec = self.gvp_layer_norm(s_vec)
        s_vec = self.ff_func(s_vec)
        s_vec = self.dropout_node(s_vec)
        s_vec = self.residual_connection_tuple(s_vec, residual_sv)
        return s_vec

