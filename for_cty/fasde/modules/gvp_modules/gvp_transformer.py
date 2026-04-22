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

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder, GraphDecoder
from .transformer_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter
from ..protein_mpnn_utils import ProteinMPNN
from ..cmlm_mask import inject_noise, _skeptical_unmasking, _skeptical_unmasking_all

from esm import pretrained
ESM_PRETRAIN = "/train14/superbrain/lhchen/protein/pretrain/esm2/params/esm2_t33_650M_UR50D.pt"


esm_dict = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
    'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V',
]
restypes = ['<unk>', '<pad>', '<cls>', '<mask>'] + restypes
restype_order = {restype: i for i, restype in enumerate(restypes)}

af2_to_esm_dict = {
    af2_i: esm_dict[restype]
    for restype, af2_i in restype_order.items()
}
sup_af2_to_esm_dict = {
    i: esm_dict['<mask>']
    for i in np.arange(len(af2_to_esm_dict), 35)
}
af2_to_esm_dict.update(sup_af2_to_esm_dict)

af2_to_esm_convert_indices = torch.zeros(len(af2_to_esm_dict))
for af2_i, esm_i in af2_to_esm_dict.items():
    af2_to_esm_convert_indices[af2_i] = esm_i
af2_to_esm_convert_indices = af2_to_esm_convert_indices.long()

# import pdb; pdb.set_trace()
import logging
logger = logging.getLogger(__name__)


def make_prev_token_from_t(time_step, T, B, L, lig_mask, device, output_scores_mask, merged_s):
    cur_iter_p = 1 - (time_step / T)
    output_scores = torch.rand((B, L)).to(device)
    skeptical_mask_prot = _skeptical_unmasking_all(output_scores.masked_fill(~output_scores_mask, 2.0), output_scores_mask, cur_iter_p)
    cur_iter_masked_S = merged_s.masked_fill(skeptical_mask_prot, 0.0)
    cur_iter_masked_S = cur_iter_masked_S.masked_fill(lig_mask.bool(), 24)

    return cur_iter_masked_S


def converter_from_af2_to_esm(pred_merged_aatype, prot_mask):
    device =pred_merged_aatype.device
    esm_no_bos_toks = []
    for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
        b_esm_tokens = af2_to_esm_convert_indices[b_af2_tokens]
        esm_no_bos_toks.append(b_esm_tokens)
    esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)

    B, Np = esm_no_bos_toks.shape
    esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0)
    esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 1)
    esm_toks[range(B), (prot_mask.sum(1)+1).long()] = 2

    return esm_toks.long(), esm_no_bos_toks.long()


def uniform_sampling_t(diffusion_steps, batch_size, device):
    w = np.ones([diffusion_steps])
    p = w / np.sum(w)
    indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
    indices = torch.from_numpy(indices_np).long().to(device)
    # weights_np = 1 / p[indices_np]
    # weights = torch.from_numpy(weights_np).float().to(device)
    return indices#, weights



class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        hidden_dim = 128

        self.proteinmpnn = ProteinMPNN(
            num_letters=35,
            node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, \
             num_encoder_layers=args.merge_mpnn_enc_layer_num, num_decoder_layers=args.merge_mpnn_dec_layer_num, augment_eps=args.augment_eps, k_neighbors=48, vocab=35, \
                nar=args.nar, lig_neighbor_seq_mask=args.lig_neighbor_seq_mask, ligmpnn_init=args.ligmpnn_init,
                pre_prot_mode=args.pre_prot_mode, freeze_encoder_param=args.freeze_encoder_param,
                esm_embedder=args.esm_embedder,
                max_iter_num=args.max_iter_num,
                embed_unimol_reprs=args.embed_unimol_reprs,
                encode_mpnn=args.encode_mpnn
                )

        self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet(ESM_PRETRAIN)
        self.T = args.diff_T

        # if self.training:
        if args.pretrained_mpnn_ckpt:
            state = torch.load(args.pretrained_mpnn_ckpt_f, map_location='cpu')
            model_weights = state["model"]
            model_weights = {k.replace('model.proteinmpnn.', ''): v for k, v in model_weights.items() \
                if (k.startswith('model.proteinmpnn.') ) }
            self.proteinmpnn.load_state_dict(
                model_weights, strict=False
            )

        self._freeze_esm_parameter()


    def _freeze_esm_parameter(self, ):
        for name, param in self.esm_model.named_parameters():
            param.requires_grad = False
        logger.info(f'freeze parameters of esm model')


    def forward(
        self,
        batch,
        use_gradient_checkpoint: bool = True,
    ):
        merged_coords = batch['coords']
        merged_s = batch['tokens']
        merged_mask = batch['node_mask']

        B, L = merged_coords.shape[:2]
        device = merged_coords.device
        merged_chain_mask = torch.ones((B,L)).to(device)
        merged_residue_idx = batch['residx']

        merged_chain_encoding_all = batch['chainidx']
        randn = torch.randn_like(merged_chain_mask)

        merged_lig_mask = batch['lig_mask']
        lig_node_attr = batch['lig_node_attr']
        lig_edge_index = batch['lig_edge_index']
        lig_edge_attr = batch['lig_edge_attr']

        if batch.__contains__('esm_embedding'):
            esm_embedding = batch['esm_embedding']
        else:
            esm_embedding = None

        if batch.__contains__('last_iter_aatype'):
            last_iter_aatype = batch['last_iter_aatype']
        else:
            last_iter_aatype = None

        unimol_reprs = batch['unimol_lig_node_attr']
        torsion_angles=batch['chi_angles']
        alt_chi_angles = batch['alt_chi_angles']
        torsion_angle_mask=batch['chi_mask']
        plip_anno_itype_list = batch['plip_anno_itype_list']
        prot_mask = (torch.all(torch.stack([(1-merged_lig_mask), merged_mask], -1), -1)).float()
        output_scores_mask = (merged_s * (1 - prot_mask)) == 0
        cur_timestep = uniform_sampling_t(self.T, B, device).long()

        if (np.random.rand() < 0.5):
            last_iter_S_embed = torch.zeros((B, L, 1280)).to(device)
        else:
            with torch.no_grad():
                last_timestep = torch.where(cur_timestep.float() > 0, cur_timestep - 1, 0).long()
                last_iter_null_embed = torch.zeros((B, L, 1280)).to(device)
                last_iter_esm_embed = torch.zeros((B, L, 1280)).to(device)
                last_prev_tokens = make_prev_token_from_t(last_timestep, self.T, B, L, merged_lig_mask, device, output_scores_mask, merged_s)

                logits, src_token, log_sc_logits, prev_torsion_mask = self.proteinmpnn(
                    merged_coords,
                    merged_s,
                    merged_mask,
                    merged_chain_mask,
                    merged_residue_idx,
                    merged_chain_encoding_all,
                    randn,
                    lig_node_attr, lig_edge_attr, lig_edge_index,
                    use_input_decoding_order=False,
                    decoding_order=None,
                    lig_mask=merged_lig_mask,
                    seq_mask=None,
                    esm_embedding=esm_embedding,
                    last_iter_S_embed=last_iter_null_embed,
                    unimol_reprs=unimol_reprs,
                    torsion_angles=torsion_angles,
                    alt_chi_angles=alt_chi_angles,
                    torsion_angle_mask=torsion_angle_mask,
                    prev_tokens=last_prev_tokens,
                    plip_anno_itype_list=plip_anno_itype_list)

                tokens_mask = last_prev_tokens == 0
                tokens_mask = tokens_mask.int()

                pred_aatype = torch.argmax(logits[..., 4: 24], -1) + 4
                pred_merged_aatype = tokens_mask * pred_aatype + (1 - tokens_mask) * merged_s
                input_esm_tokens, raw_esm_tokens = converter_from_af2_to_esm(pred_merged_aatype, prot_mask)
                output = self.esm_model(input_esm_tokens, repr_layers=[33], return_contacts=False)
                esm_rep_ = output['representations'][33][:, 1:-1].detach()

            last_iter_esm_embed = torch.where((prot_mask == 1)[..., None], esm_rep_, last_iter_esm_embed)
            last_iter_S_embed = torch.where((cur_timestep == 0)[:, None, None], last_iter_null_embed, last_iter_esm_embed)

        cur_prev_tokens = make_prev_token_from_t(cur_timestep, self.T, B, L, merged_lig_mask, device, output_scores_mask, merged_s)

        logits, src_token, log_sc_logits, prev_torsion_mask = self.proteinmpnn(
            merged_coords, merged_s, merged_mask, merged_chain_mask, merged_residue_idx, merged_chain_encoding_all, randn,
            lig_node_attr, lig_edge_attr, lig_edge_index,
            use_input_decoding_order=False, decoding_order=None,
            lig_mask=merged_lig_mask, seq_mask=None,esm_embedding=esm_embedding, last_iter_S_embed=last_iter_S_embed,
            unimol_reprs=unimol_reprs, torsion_angles=torsion_angles, alt_chi_angles=alt_chi_angles, torsion_angle_mask=torsion_angle_mask,
            prev_tokens=cur_prev_tokens, plip_anno_itype_list=plip_anno_itype_list)


        return logits, src_token, log_sc_logits, prev_torsion_mask


    def sample(
        self,
        batch,
        prev_output_tokens = None,
        target_shuffle_index: Tensor = None,
        target_inv_shuffle_index: Tensor = None,
        return_all_hiddens: bool = False,
        features_only: bool = False,
        use_gradient_checkpoint: bool = True,
        tmp = 0.1
    ):
        coords = batch['coords']
        confidence = batch['confidence']
        positions = batch['residx']
        padding_mask = batch['padding_mask'] = (batch['node_mask'] == 0.0)
        encoder_out = self.encoder(batch, coords, padding_mask, confidence, positions,
            return_all_hiddens=return_all_hiddens,use_gradient_checkpoint=use_gradient_checkpoint)
        max_len = batch['seg_len'][:,0].max()
        lig_mask = batch['lig_mask'].bool()
        rec_mask = batch['node_mask'] * (1 - lig_mask.float())
        # prev_output_tokens = (prev_output_tokens*rec_mask) [:,:max_len-1].long()
        gg = prev_output_tokens*rec_mask + (1-rec_mask)*torch.ones_like(prev_output_tokens)
        prev_output_tokens = gg[:,:max_len-1].long()
        positions = (positions*rec_mask)[:,:max_len].long()
        target_shuffle_index = (target_shuffle_index*rec_mask)[:,:max_len].long()
        extra = None
        x = encoder_out['encoder_out'][0]
        edge_index = encoder_out['edge_index']
        edge_embeddings = encoder_out['edge_embeddings']
        L = batch['seg_len'][0,0]
        tokens = torch.tensor([[1,33]]).long().to(x.device)

        for i in range(2,L):
            x_ = x[:,:i]
            emb_s = self.embed_s(tokens)
            emb_s = torch.concat([emb_s[:,-1:], emb_s[:,:-1]],dim=1)
            x_ = torch.concat([emb_s, x_], dim=-1)
            mask = (edge_index[0]<i)&(edge_index[1]<i)
            edge_index_ = edge_index[:,mask]
            edge_embeddings_ = edge_embeddings[mask, :]
            x_ = self.decoder(x_,  edge_index_, edge_embeddings_)
            logits = self.cls(x_)
            if tmp<=0:
                token = logits[:,-1].argmax()
            else:
                probs = F.softmax(logits[:,-1]/tmp, dim=-1)
                token = torch.multinomial(probs, 1)[0,0]
            tokens = torch.concat([tokens, token[None,None]],dim=-1)
        xx =  batch['tokens'][0,1:L-1]==tokens[0,2:]
        # print(xx.float().mean())
        return tokens[:,2:]

    # def sample(
    #     self,
    #     batch,
    #     batch_coords,
    #     confidence,
    #     padding_mask,
    #     positions,
    #     partial_seq=None,
    #     temperature=1.0,
    #     target_shuffle_index=None,
    #     target_inv_shuffle_index=None,
    #     num_refine_iteration = 0,
    #     return_probs=True
    # ):
    #     """
    #     Samples sequences based on multinomial sampling (no beam search).

    #     Args:
    #         coords: L x 3 x 3 list representing one backbone
    #         partial_seq: Optional, partial sequence with mask tokens if part of
    #             the sequence is known
    #         temperature: sampling temperature, use low temperature for higher
    #             sequence recovery and high temperature for higher diversity
    #         confidence: optional length L list of confidence scores for coordinates
    #     """
    #     # L = batch_coords.shape[1] - 2
    #     L = batch['seg_len'][0,0] - 1
    #     # Convert to batch format
    #     # batch_converter = CoordBatchConverter(self.decoder.dictionary)
    #     # batch_coords, confidence, _, _, padding_mask = (
    #     #     batch_converter([(coords.cpu(), confidence, None)])
    #     # )
    #     # Start with prepend token
    #     # print('sampling')
    #     mask_idx = self.decoder.dictionary.get_idx('<mask>')
    #     sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int).to(batch_coords.device)
    #     sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
    #     if partial_seq is not None:
    #         # import pdb; pdb.set_trace()
    #         for i, c in enumerate(partial_seq):
    #             sampled_tokens[0, i+1] = self.decoder.dictionary.get_idx(c)

    #     # Save incremental states for faster sampling
    #     incremental_state = dict()

    #     # Run encoder only once
    #     encoder_out = self.encoder(batch, batch_coords, padding_mask, confidence, positions)

    #     # Decode one token at a time
    #     if return_probs:
    #         all_probs=[]
    #     for i in range(1, L+1):
    #         if sampled_tokens[0, i] != mask_idx:
    #             continue
    #         logits, _ = self.decoder(
    #             sampled_tokens[:, :i],
    #             positions,
    #             encoder_out = encoder_out,
    #             incremental_state=incremental_state,
    #             target_shuffle_index = target_shuffle_index,
    #             is_sampling = True,
    #             batch = batch
    #         )
    #         logits = logits[0].transpose(0, 1)
    #         if temperature <= 0.0:
    #             # max sample
    #             sampled_tokens[:, i] = torch.argmax(logits, dim=-1)
    #             probs = F.softmax(logits, dim=-1)
    #             if return_probs:
    #                 all_probs.append(probs)
    #         else:
    #             logits /= temperature
    #             probs = F.softmax(logits, dim=-1)
    #             # import pdb;pdb.set_trace
    #             # print(probs.shape,probs)
    #             if return_probs:
    #                 all_probs.append(probs)


    #             sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
    #     if return_probs:
    #         probs=np.concatenate([logits[:, 4:24].cpu().numpy() for logits in all_probs])
    #     restypes = [
    #         'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    #         'S', 'T', 'W', 'Y', 'V','X','X'
    #     ]
    #     dictionary = {i:v for  i,v in enumerate(restypes) }
    #     def ids_to_sequence(ids):
    #         return ''.join([dictionary[a.item()] for a in sampled_tokens[0, 1:-1]])
    #     sequence_trajectory = [ ids_to_sequence(sampled_tokens[0, 1:-1]) ]
    #     for niter in range(num_refine_iteration):
    #         sampled_tokens = self.decoder.refine(
    #             sampled_tokens, positions[:, :-1], temperature=temperature,
    #             encoder_out=encoder_out
    #         )
    #         sequence_trajectory.append(ids_to_sequence(sampled_tokens[0, 1:]))

    #     # sampled_seq = sampled_tokens[0, 1:]

    #     # # Convert back to string via lookup
    #     # return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])
    #     if return_probs:
    #         return sequence_trajectory,probs
    #     return sequence_trajectory
