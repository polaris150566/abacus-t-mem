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
from scipy.spatial import transform

from ..data import Alphabet

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_progen_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter 


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
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

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
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        positions,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence, positions,
            return_all_hiddens=return_all_hiddens)
        logits, extra, lm_output = self.decoder(
            prev_output_tokens,
            positions[:, :-1],
            lm_past_key_values = None,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )

        token_id_start = 4
        token_ids = torch.arange(4, 24).to(coords.device)
        lm_token_ids = torch.LongTensor([self.decoder.esm_to_progen_ids[idx.item()] for idx in token_ids]).to(coords.device)

        dec_logits = logits[0].transpose(0, 1)
        dec_logits = torch.index_select(dec_logits, -1, token_ids)

        lm_logits = lm_output.logits[0]
        lm_logits = torch.index_select(lm_logits, -1, lm_token_ids)


        return logits, extra, dec_logits, lm_logits
    
    def sample(
        self, 
        batch_coords, 
        confidence,
        padding_mask, 
        positions,
        partial_seq=None, 
        temperature=1.0, 
        lm_weight=0.5, 
        lm_temperature=1.0, 
        lm_weight_warmup=0,
    ):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = batch_coords.shape[1] - 2
        # Convert to batch format
        # batch_converter = CoordBatchConverter(self.decoder.dictionary)
        # batch_coords, confidence, _, _, padding_mask = (
        #     batch_converter([(coords.cpu(), confidence, None)])
        # )
        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int).to(batch_coords.device)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i+1] = self.decoder.dictionary.get_idx(c)
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        # incremental_state=None
        # import pdb; pdb.set_trace()
        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence, positions)

        lm_past_key_values = None

        token_id_start = 4
        token_ids = torch.arange(4, 24).to(batch_coords.device)
        lm_token_ids = torch.LongTensor([self.decoder.esm_to_progen_ids[idx.item()] for idx in token_ids]).to(batch_coords.device)

        sample_probs = []
        sample_lm_probs = []
        
        # Decode one token at a time
        for i in range(1, L+1):
            if sampled_tokens[0, i] != mask_idx:
                continue
            # import pdb; pdb.set_trace()
            logits, _, lm_output = self.decoder(
                sampled_tokens[:, :i], 
                positions,
                lm_past_key_values,
                encoder_out = encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits = torch.index_select(logits, -1, token_ids)
            if temperature > 0.0:
                logits /= temperature
            probs = F.softmax(logits, dim=-1)

            lm_logits = lm_output.logits[0]
            lm_logits = torch.index_select(lm_logits, -1, lm_token_ids)
            if lm_temperature > 0.0:
                lm_logits /= lm_temperature
            lm_probs = F.softmax(lm_logits, dim=-1)

            lm_past_key_values = lm_output.past_key_values

            lm_weight_t = lm_weight if i >= lm_weight_warmup else float(i) / float(lm_weight_warmup) * lm_weight

            mix_probs = (1-lm_weight_t) * probs + lm_weight_t * lm_probs
            if temperature <= 0.0 or lm_temperature <= 0.0:
                sample_id = torch.argmax(mix_probs, dim=-1)
            else:
                sample_id = torch.multinomial(mix_probs, 1).squeeze(-1)
            sampled_tokens[:, i] = sample_id + token_id_start

            sample_probs.append(
                # torch.gather(probs, -1, sample_id[:, None])
                probs,
            )
            sample_lm_probs.append(
                # torch.gather(lm_probs, -1, sample_id[:, None])
                lm_probs,
            )
        sampled_seq = sampled_tokens[0, 1:]

        sample_probs = torch.cat(sample_probs, dim=0)
        sample_lm_probs = torch.cat(sample_lm_probs, dim=0)

        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq]), {'if_probs': sample_probs.cpu().numpy(), 'lm_probs': sample_lm_probs.cpu().numpy()}

