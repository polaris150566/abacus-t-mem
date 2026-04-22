# Contents of this file are from the open source code for
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

import typing as T
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter



def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    # clamp is slow
    # out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    out = torch.sum(torch.square(x), axis, keepdims) + eps
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)



class Conv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture
    '''
    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3,
            vector_gate=False, module_list=None, aggr="mean", eps=1e-8,
            activations=(F.relu, torch.sigmoid)):
        super(Conv, self).__init__(aggr=aggr)
        self.eps = eps
        self.si = in_dims
        self.so = out_dims
        self.se = edge_dims
        
        module_list = module_list or []
        if not module_list:
            # module_list.append(
            #     nn.Linear(2*self.si + self.se, self.so)
            # )
            if n_layers == 1:
                module_list.append(
                    nn.Linear(2*self.si + self.se, self.so))
            else:
                module_list.append(
                    nn.Linear(2*self.si + self.se, self.so)
                )
                for i in range(n_layers):
                    module_list.append(nn.Linear(self.so, self.so))
                
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s = x
        message = self.propagate(edge_index, 
                    s=x_s, 
                    edge_attr=edge_attr)
        return message

    def message(self, s_i, s_j, edge_attr):
        message = torch.concat((s_j, edge_attr, s_i), dim=-1)
        message = self.message_func(message)
        return message


class ConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    '''
    def __init__(self, node_dims, edge_dims, vector_gate=False,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, attention_heads=0,
                 conv_activations=(F.relu, torch.sigmoid),
                 n_edge_gvps=0, layernorm=True, eps=1e-8):
        
        super(ConvLayer, self).__init__()
        if attention_heads == 0:
            self.conv = Conv(
                    node_dims, node_dims, edge_dims, n_layers=n_message,
                    vector_gate=vector_gate,
                    aggr="add" if autoregressive else "mean",
                    activations=conv_activations, 
                    eps=eps,
            )
        else:
            raise NotImplementedError
        if layernorm:
            self.norm = nn.ModuleList([nn.LayerNorm(node_dims, eps=eps) for _ in range(2)])
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(nn.Linear(node_dims, node_dims, ))
        else:
            hid_dims = 4*node_dims
            ff_func.append(nn.Linear(node_dims, hid_dims,))
            for i in range(n_feedforward-2):
                ff_func.append(nn.Linear(hid_dims, hid_dims, ))
            ff_func.append(nn.Linear(hid_dims, node_dims, ))
        self.ff_func = nn.Sequential(*ff_func)

        self.edge_message_func = None
        if n_edge_gvps > 0:
            si = node_dims
            se = edge_dims
            module_list = [
                nn.Linear(2*si + se, edge_dims)
            ]
            for i in range(n_edge_gvps - 2):
                module_list.append(nn.Linear(edge_dims, edge_dims,
                    vector_gate=vector_gate))
            if n_edge_gvps > 1:
                module_list.append(nn.Linear(edge_dims, edge_dims,))
            self.edge_message_func = nn.Sequential(*module_list)
            if layernorm:
                self.edge_norm = nn.LayerNorm(edge_dims, eps=eps)
            else:
                self.edge_norm = nn.Identity()
            self.edge_dropout = nn.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        if self.edge_message_func:
            src, dst = edge_index
            if autoregressive_x is None:
                x_src = x[0][src], x[1][src]
            else: 
                mask = (src < dst).unsqueeze(-1)
                x_src = (
                    torch.where(mask, x[0][src], autoregressive_x[0][src]),
                    torch.where(mask.unsqueeze(-1), x[1][src],
                        autoregressive_x[1][src])
                )
            x_dst = x[0][dst], x[1][dst]
            x_edge = (
                torch.cat([x_src[0], edge_attr[0], x_dst[0]], dim=-1),
                torch.cat([x_src[1], edge_attr[1], x_dst[1]], dim=-2)
            )
            edge_attr_dh = self.edge_message_func(x_edge)
            edge_attr = self.edge_norm(tuple_sum(edge_attr,
                self.edge_dropout(edge_attr_dh)))
        autoregressive_x = True
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src <= dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = edge_attr[mask]
            edge_attr_backward = edge_attr[~mask]
            dh = self.conv(x, edge_index_forward, edge_attr_forward)
            count = scatter_add(torch.ones_like(dst), dst,dim_size=dh.size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh / count
        else:
            dh = self.conv(x, edge_index, edge_attr)
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        x = self.norm[0]( x + self.dropout[0](dh))   
        
        dh = self.ff_func(x)
        x = self.norm[1]((x + self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[node_mask] = x
            x = x_

        return x, edge_attr
