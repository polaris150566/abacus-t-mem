# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..modules import SinusoidalPositionalEmbeddingv1
from .transformer_layer import TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint



def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
    ):
        super().__init__()
        self.args = args
        self.dictionary = dictionary
        self._future_mask = torch.empty(0)
        self._refine_mask = torch.empty(0)

        self.dropout_module = nn.Dropout(args.dropout)

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim

        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.project_in_dim = (
            nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = SinusoidalPositionalEmbeddingv1(
            embed_dim,
            self.padding_idx,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.enc_concat_projection = nn.Linear(embed_dim*2, embed_dim)
        self.enc_concat_layer_norm = nn.LayerNorm(embed_dim)

        self.build_output_projection(args, dictionary)

    def build_output_projection(self, args, dictionary):
        self.output_projection = nn.Linear(
            args.decoder_embed_dim, len(dictionary), bias=False
        )
        nn.init.normal_(
            self.output_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
        )

    def build_decoder_layer(self, args):
        return TransformerDecoderLayer(args)

    def forward(
        self,
        prev_output_tokens,
        positions,
        target_shuffle_index: Tensor = None,
        # target_inv_shuffle_index: Tensor = None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        return_all_hiddens: bool = False,
        is_sampling: bool = False,
        use_gradient_checkpoint: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if is_sampling:
            x, extra = self.extract_features_sample(
                prev_output_tokens,
                positions,
                target_shuffle_index = target_shuffle_index,
                # target_inv_shuffle_index = target_inv_shuffle_index, 
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        else:
            x, extra = self.extract_features(
                prev_output_tokens,
                positions,
                target_shuffle_index = target_shuffle_index,
                # target_inv_shuffle_index = target_inv_shuffle_index, 
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                use_gradient_checkpoint = use_gradient_checkpoint,
            )

        if not features_only:
            x = self.output_layer(x)
        x = x.transpose(1, 2) # B x T x C -> B x C x T
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        positions,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        target_shuffle_index: Tensor = None,
        # target_inv_shuffle_index: Tensor = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        use_gradient_checkpoint = False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        # embed positions
        positions = self.embed_positions(
            # prev_output_tokens
            positions
        )
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)
        if target_shuffle_index is not None:
            # assert(target_inv_shuffle_index is not None)
            x = torch.gather(x, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

            enc_shuffle = torch.transpose(enc, 0, 1)
            enc_shuffle = torch.gather(enc_shuffle, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, enc_shuffle.shape[-1]))
            enc_shuffle = torch.cat(
                [enc_shuffle[:, 1:], enc[-1:].transpose(0, 1)], dim=1
            )

            x = torch.cat([x, enc_shuffle], dim=-1)
            x = self.enc_concat_projection(x)
            x = self.enc_concat_layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if use_gradient_checkpoint and self.training:
                x, layer_attn, _ = checkpoint(
                    layer, 
                    x, enc, padding_mask, incremental_state,
                    None, None, self_attn_mask, self_attn_padding_mask,
                    False, False
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=False,
                    need_head_weights=False,
                )
            # x, layer_attn, _ = layer(
            #     x,
            #     enc,
            #     padding_mask,
            #     incremental_state,
            #     self_attn_mask=self_attn_mask,
            #     self_attn_padding_mask=self_attn_padding_mask,
            #     need_attn=False,
            #     need_head_weights=False,
            # )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1)
        # if target_shuffle_index is not None:
        #     x = torch.gather(x, 1, target_inv_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        return x, {"inner_states": inner_states}

    def extract_features_sample(
        self,
        prev_output_tokens,
        positions,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        target_shuffle_index: Tensor = None,
        # target_inv_shuffle_index: Tensor = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        # import pdb; pdb.set_trace()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        enc_shuffle = torch.transpose(enc, 0, 1)
        enc_shuffle = torch.gather(enc_shuffle, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, enc_shuffle.shape[-1]))
        enc_shuffle = torch.cat(
            [enc_shuffle[:, 1:], enc[-1:].transpose(0, 1)], dim=1
        )
        enc_shuffle = enc_shuffle[:, :slen]

        position_shuffle = torch.gather(positions, 1, target_shuffle_index)

        # embed positions
        positions = self.embed_positions(
            # prev_output_tokens
            position_shuffle[:, :slen]
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]
            enc_shuffle = enc_shuffle[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)
        # if target_shuffle_index is not None:
            # assert(target_inv_shuffle_index is not None)
            # x = torch.gather(x, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        x = torch.cat([x, enc_shuffle], dim=-1)
        x = self.enc_concat_projection(x)
        x = self.enc_concat_layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1)
        # if target_shuffle_index is not None:
        #     x = torch.gather(x, 1, target_inv_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        return x, {"inner_states": inner_states}

    def extract_features_refine(
        self,
        prev_output_tokens,
        positions,
        encoder_out: Optional[Dict[str, List[Tensor]]],
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        # embed positions
        positions = self.embed_positions(
            # prev_output_tokens
            positions
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)


        # if target_shuffle_index is not None:
        #     # assert(target_inv_shuffle_index is not None)
        #     x = torch.gather(x, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        #     enc_shuffle = torch.transpose(enc, 0, 1)
        #     enc_shuffle = torch.gather(enc_shuffle, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, enc_shuffle.shape[-1]))
        #     enc_shuffle = torch.cat(
        #         [enc_shuffle[:, 1:], enc[-1:].transpose(0, 1)], dim=1
        #     )
        x = torch.cat([x, enc[1:, :].transpose(0, 1)], dim=-1)
        x = self.enc_concat_projection(x)
        x = self.enc_concat_layer_norm(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        for idx, layer in enumerate(self.layers):
            self_attn_mask = self.buffered_refine_mask(x) # self.buffered_future_mask(x)

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state = None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)

        return x, {"inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def buffered_refine_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._refine_mask.size(0) == 0
            or (not self._refine_mask.device == tensor.device)
            or self._refine_mask.size(0) < dim
        ):
            # self._refine_mask = torch.triu(
            #     fill_with_neg_inf(torch.zeros([dim, dim])), 1
            # )
            self._refine_mask = torch.zeros((dim, dim)) 
            for i in range(dim-1):
                self._refine_mask[i, i+1] = - torch.inf
        self._refine_mask = self._refine_mask.to(tensor)
        return self._refine_mask[:dim, :dim]

    def refine(
        self,
        prev_output_tokens,
        positions,
        temperature=1.0,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features_refine(
            prev_output_tokens,
            positions,
            encoder_out=encoder_out,
        )

        # x = x.transpose(1, 2) # B x T x C -> B x C x T
        logits = x = self.output_layer(x)
        if temperature <= 0.0:
            sampled_tokens = torch.argmax(logits, dim=-1)
        else:
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)

        sampled_tokens = torch.cat(
            [prev_output_tokens[:, :1], sampled_tokens[:, :-1]],
            dim = 1
        )

        return sampled_tokens
