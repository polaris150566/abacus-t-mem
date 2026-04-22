# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..modules import SinusoidalPositionalEmbeddingv1
from ..constants import proteinseq_toks
from .transformer_layer import TransformerDecoderLayer


from tokenizers import Tokenizer
from ..progen.modeling_progen import ProGenForCausalLM


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

    def load_pretrained_progen(self, pretrained_path):
        def create_tokenizer_custom(file):
            with open(file, 'r') as f:
                return Tokenizer.from_str(f.read())
        self.progen_tokenizer = create_tokenizer_custom(file=f'{pretrained_path}/tokenizer.json')
        self.progen = ProGenForCausalLM.from_pretrained(f'{pretrained_path}')

    def load_pfam_lm(self, pfam_lm_path):
        ckpt = torch.load(pfam_lm_path, map_location='cpu')
        state_dict = {re.sub(r'^model\.', '', k): v for k, v in ckpt['model'].items()}
        self.progen.load_state_dict(state_dict)

    def make_token_convertor(self, ):
        self.esm_to_progen_ids = { self.dictionary.get_idx(tok): self.progen_tokenizer.encode(tok).ids[0] for tok in proteinseq_toks['toks'][:20]}
        self.esm_to_progen_ids[self.dictionary.get_idx('<cath>')] = self.progen_tokenizer.encode('1').ids[0]
        self.progen_to_esm_ids = { v: k for k, v in self.esm_to_progen_ids.items()}

    def forward(
        self,
        prev_output_tokens,
        positions,
        lm_past_key_values,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        return_all_hiddens: bool = False,
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
        x, extra, lm_output = self.extract_features(
            prev_output_tokens,
            positions,
            lm_past_key_values,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )

        if not features_only:
            x = self.output_layer(x)
        x = x.transpose(1, 2) # B x T x C -> B x C x T
        return x, extra, lm_output

    def extract_features(
        self,
        prev_output_tokens,
        positions,
        lm_past_key_values,
        encoder_out: Optional[Dict[str, List[Tensor]]],
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

        enc_representation = torch.transpose(enc, 0, 1)
        enc_representation = enc_representation[:, 1:slen+1]

        # enc_shuffle = torch.transpose(enc, 0, 1)
        # enc_shuffle = torch.gather(enc_shuffle, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, enc_shuffle.shape[-1]))
        # enc_shuffle = torch.cat(
        #     [enc_shuffle[:, 1:], enc[-1:].transpose(0, 1)], dim=1
        # )
        # enc_shuffle = enc_shuffle[:, :slen]

        # position_shuffle = torch.gather(positions, 1, target_shuffle_index)

        # embed positions
        positions = self.embed_positions(
            # prev_output_tokens
            positions[:, :slen]
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]
            enc_representation = enc_representation[:, -1:]

        lm_input_ids = [[self.esm_to_progen_ids[idx.item()] for idx in toks] for toks in prev_output_tokens]
        lm_input_ids = torch.LongTensor(lm_input_ids).to(prev_output_tokens.device)
        lm_input_ids = torch.reshape(lm_input_ids, prev_output_tokens.shape)

        lm_attention_mask = torch.ones((bs, slen)).long().to(prev_output_tokens.device)
        lm_position_ids = torch.zeros((bs, 1)).long().to(prev_output_tokens.device) + slen - 1

        import pdb; pdb.set_trace()

        lm_output = self.progen(
            input_ids = lm_input_ids,
            past_key_values = lm_past_key_values,
            attention_mask = lm_attention_mask,
            position_ids = lm_position_ids,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = True,
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        x += positions

        x = self.dropout_module(x)
        # if target_shuffle_index is not None:
            # assert(target_inv_shuffle_index is not None)
            # x = torch.gather(x, 1, target_shuffle_index.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        x = torch.cat([x, enc_representation], dim=-1)
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

        return x, {"inner_states": inner_states}, lm_output

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
