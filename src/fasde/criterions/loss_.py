import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import lengths_to_mask
from fairseq.dataclass import FairseqDataclass
from .losses import (

    score_matching_loss,
    cent_loss,
    ligand_bond_loss,
    ligand_angle_loss,
    ligand_torsion_loss,
    protein_angle_loss,
    protein_bond_loss,
    protein_torsion_loss,
    protein_phipsi_loss,
    protein_sidechain_loss,
    reciprocal_loss
)

import sys
sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
from alphafold.common import residue_constants

logger = logging.getLogger(__name__)

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


def discrete_torsion(dist, edges=None):
    min_, max_, nbin_ = edges
    dist = (dist - min_) * nbin_ / (max_ - min_)
    dist = dist.int()
    dist = torch.clip(dist, 0, nbin_-1)
    return dist


@dataclass
class CriterionConfig(FairseqDataclass):
    loss_reduction: str = field(default="sum", metadata={"help": "loss reduction"})


@register_criterion("diff_full_atom", dataclass=CriterionConfig)
class DiffFullAtomCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.loss_reduction = 'sum'


    def converter_from_af2_to_esm(self, pred_merged_aatype, rec_len, prot_mask):
        device =pred_merged_aatype.device
        esm_no_bos_toks = []
        for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
            b_esm_tokens = af2_to_esm_convert_indices[b_af2_tokens]
            esm_no_bos_toks.append(b_esm_tokens)

        esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)
        esm_no_bos_toks = (1 - prot_mask) + prot_mask * esm_no_bos_toks

        esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0)
        esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 2)
        # for b_idx, b_prot_len in enumerate(rec_len):
        #     esm_toks[b_idx][b_prot_len.item()+1] = 2

        return esm_toks.long()


    def _forward_iter_step(
        self, model, sample, nar, pred_lig_neighbor_seq_mask):
        logits, src_token = model(sample)

        logits_f = logits.float()
        tgt_token = sample['tokens']
        res_num = tgt_token.shape[-1]
        lig_neighbor_seq_mask = sample['lig_neighbor_seq_mask']

        lig_mask = sample['lig_mask']
        node_mask = sample['node_mask']
        # import pdb; pdb.set_trace()
        prot_mask = (torch.all(torch.stack([(1-lig_mask), node_mask], -1), -1)).float()

        if not nar:
            if pred_lig_neighbor_seq_mask:
                tokens_mask = torch.all(torch.stack([lig_neighbor_seq_mask, prot_mask], -1), -1)
            else:
                tokens_mask = prot_mask
            tokens_mask = tokens_mask.int()
        else:
            tokens_mask = src_token == 0
            tokens_mask = tokens_mask.int()

        logits_f = logits.float()

        loss_tgt_token = tgt_token * tokens_mask
        logits = logits * tokens_mask[..., None]

        logits_f = logits.float()
        sample_size = tokens_mask.sum()
        batch_size = logits.shape[0]

        loss = F.cross_entropy(
            torch.reshape(logits_f, (-1, logits_f.size(-1))),
            torch.reshape(loss_tgt_token, (-1,)),
            reduction = 'none'
            ).reshape(loss_tgt_token.shape)

        reduced_loss = (loss * tokens_mask).sum()/(tokens_mask.sum() + 1e-6)

        aatype_pred = torch.argmax(logits_f, dim=-1)
        #生成预测的氨基酸
        aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask)/(tokens_mask.sum() + 1e-6)

        pred_merged_aatype = tokens_mask * aatype_pred + (1 - tokens_mask) * tgt_token

        pred_merged_aatype_esm = self.converter_from_af2_to_esm(
            pred_merged_aatype, sample['rec_len'], prot_mask)

        esmmodel = model.model.proteinmpnn.esm_model
        repr_layers = [esmmodel.num_layers]
        esm_out = esmmodel(pred_merged_aatype_esm.long(), repr_layers=repr_layers)

        representations = list(esm_out["representations"].values())[-1]
        representations = representations[:, 1:res_num+1]
        return representations, reduced_loss, aatype_acc, sample_size, batch_size, logits, aatype_pred


    def forward_iter(self, model, sample):
        nar = model.model.proteinmpnn.nar
        pred_lig_neighbor_seq_mask = model.model.proteinmpnn.lig_neighbor_seq_mask
        max_iter_num = model.model.proteinmpnn.max_iter_num
        traj_loss_weight = model.model.proteinmpnn.traj_loss_weight
        ### esm embedding recycle ###

        iter_num = np.random.choice(np.arange(max_iter_num))+1
        loss = []
        acc = []

        for cur_iter in range(iter_num):
            cur_representations, cur_loss, aatype_acc, sample_size, batch_size, logits, cur_aatype = \
                self._forward_iter_step(model, sample, nar, pred_lig_neighbor_seq_mask)
            acc.append(aatype_acc)
            loss.append(cur_loss)
            sample['esm_embedding'] = cur_representations
            sample['last_iter_aatype'] = cur_aatype

        loss = torch.stack(loss)
        acc = torch.stack(acc)

        last_loss = loss[-1]
        last_acc = acc[-1]
        if (iter_num != 1):
            traj_loss = loss.mean()
            traj_acc = acc.mean()
            loss = last_loss + traj_loss_weight * traj_loss
        else:
            traj_loss = last_loss
            traj_acc = last_acc
            loss = last_loss
        logging_output = {
            'loss': loss,
            'traj_loss': traj_loss,
            'traj_acc': traj_acc,
            'last_loss': last_loss,
            'last_acc': last_acc,
            'sample_size': sample_size,
            'ntokens': sample_size,
            'nsentences': batch_size,
        }

        loss = loss.type_as(logits)
        return loss, sample_size, logging_output


    def forward(self, model, sample):

        nar = model.model.proteinmpnn.nar
        pred_lig_neighbor_seq_mask = model.model.proteinmpnn.lig_neighbor_seq_mask
        max_iter_num = model.model.proteinmpnn.max_iter_num
        esm_embedder = model.model.proteinmpnn.esm_embedder
        torsion_bins = model.model.proteinmpnn.torsion_bins
        unavail_torsion_loss_weight = model.model.proteinmpnn.torsion_loss_weight
        avail_torsion_loss_weight = model.model.proteinmpnn.avail_torsion_loss_weight

        if esm_embedder:
            loss, sample_size, logging_output = self.forward_iter(model, sample)
            return loss, sample_size, logging_output

        else:
            logits, src_token, log_sc_logits, avail_torsion_mask = model(sample)
            ignore_index = 0
            logits_f = logits.float()
            log_sc_logits_f = log_sc_logits.float()

            tgt_token = sample['tokens']
            chi_angles = sample['chi_angles']
            disc_chi_angles = discrete_torsion(chi_angles, (-np.pi, np.pi, torsion_bins))
            alt_chi_angles = sample['alt_chi_angles']
            disc_alt_chi_angles = discrete_torsion(alt_chi_angles, (-np.pi, np.pi, torsion_bins))
            torsion_angle_mask = sample['chi_mask']
            unavail_torsion_mask = torsion_angle_mask - avail_torsion_mask

            lig_mask = sample['lig_mask']
            node_mask = sample['node_mask']

            prot_mask = (torch.all(torch.stack([(1-lig_mask), node_mask], -1), -1)).float()
            torsion_angle_mask = torsion_angle_mask * prot_mask[..., None]

            ## residue_type_one_hot = F.one_hot((sample['aatype'] * prot_mask), residue_constants.restype_num + 1)[None].float().to(sample['aatype'].device)
            ## chi_pi_periodic = torch.einsum('...ijk, ...kl->...ijl', residue_type_one_hot,torch.FloatTensor(residue_constants.chi_pi_periodic).to(torsion_angle_mask.device))

            if not nar:
                if pred_lig_neighbor_seq_mask:
                    tokens_mask = torch.all(torch.stack([prot_mask], -1), -1)
                else:
                    tokens_mask = prot_mask
                tokens_mask = tokens_mask.int()
            else:
                tokens_mask = src_token == 0
                tokens_mask = tokens_mask.int()

            ########################################################
            tokens_mask = (node_mask * (1-lig_mask)).int()
            ########################################################

            tgt_token = tgt_token * tokens_mask
            logits = logits * tokens_mask[..., None]

            logits_f = logits.float()
            sample_size = tokens_mask.sum()
            batch_size = logits.shape[0]

            ca_coords = sample['coords'][:, :, 1]
            lig_ca_coords = sample['lig_mask'][..., None] * ca_coords
            prot_ca_coords = ((1 - sample['lig_mask']) * sample['node_mask'])[..., None] * ca_coords
            all_atom_dist = torch.sqrt(torch.sum((prot_ca_coords[:, :, None] - lig_ca_coords[:, None])**2, -1)) # B x Np+Nl x Np+Nl
            all_atom_dist_mask = ((1 - sample['lig_mask']) * sample['node_mask'])[:, :, None] * sample['lig_mask'][:, None]
            all_atom_dist = all_atom_dist + (1 - all_atom_dist_mask)* 1e6
            lig_neighbor_mask = torch.any(all_atom_dist < 10.0, dim=-1).long()
            pocket_weight = 0.5
            lig_neighbor_weight = (lig_neighbor_mask * pocket_weight + ((1 - sample['lig_mask']) * sample['node_mask']))

            loss = F.cross_entropy(
                torch.reshape(logits_f, (-1, logits_f.size(-1))),
                torch.reshape(tgt_token, (-1,)), reduction = 'none').reshape(tgt_token.shape)
            loss = loss * lig_neighbor_weight
            aatype_reduced_loss = (loss * tokens_mask).sum()/(tokens_mask.sum() + 1e-6)

            aatype_pred = torch.argmax(logits_f, dim=-1)
            aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask)/(tokens_mask.sum() + 1e-6)
            # DEBUG: aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask, -1)/(tokens_mask.sum(-1) + 1e-6)

            # import pdb; pdb.set_trace()
            torsion_loss = F.cross_entropy(
                torch.reshape(log_sc_logits_f, (-1, log_sc_logits_f.size(-1))),
                torch.reshape(disc_chi_angles.long(), (-1,)), reduction = 'none').reshape(disc_chi_angles.shape)

            alt_torsion_loss = F.cross_entropy(
                torch.reshape(log_sc_logits_f, (-1, log_sc_logits_f.size(-1))),
                torch.reshape(disc_alt_chi_angles.long(), (-1,)), reduction = 'none').reshape(disc_alt_chi_angles.shape)

            torsion_loss = torch.min(torch.stack([torsion_loss, alt_torsion_loss]), dim=0)[0]
            torsion_loss = torsion_loss * lig_neighbor_weight[..., None]
            avail_reduced_torsion_loss = (torsion_loss * avail_torsion_mask).sum()/(avail_torsion_mask.sum() + 1e-6)
            unavail_reduced_torsion_loss = (torsion_loss * unavail_torsion_mask).sum()/(unavail_torsion_mask.sum() + 1e-6)

            torsion_pred = torch.argmax(log_sc_logits_f, dim=-1)
            torsion_acc = (torsion_pred == disc_chi_angles).float() * torsion_angle_mask
            alt_torsion_acc = (torsion_pred == disc_alt_chi_angles).float() * torsion_angle_mask
            torsion_acc = torch.max(torch.stack([torsion_acc, alt_torsion_acc]), dim=0)[0]
            reduced_torsion_acc = (torsion_acc * torsion_angle_mask).sum()/(torsion_angle_mask.sum() + 1e-6)

            reduced_torsion_loss = avail_reduced_torsion_loss * avail_torsion_loss_weight + unavail_reduced_torsion_loss * unavail_torsion_loss_weight
            reduced_loss = aatype_reduced_loss + reduced_torsion_loss

            logging_output = {
                'loss': reduced_loss,
                'aatype_loss': aatype_reduced_loss,
                'aatype_acc': aatype_acc,
                'torsion_loss': reduced_torsion_loss,
                'torsion_acc': reduced_torsion_acc,
                'sample_size': sample_size,
                'ntokens': sample_size,
                'nsentences': batch_size,
            }
            reduced_loss = reduced_loss.type_as(logits)
            return reduced_loss, sample_size, logging_output


    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) :
        """Aggregate logging outputs from data parallel training."""
        node_num = len(logging_outputs)
        if node_num == 0:
            node_num = 1
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("sample_size", sample_size, sample_size, round=4)

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / node_num, 1, round=4)

        aatype_loss_sum = sum(log.get("aatype_loss", 0) for log in logging_outputs)
        metrics.log_scalar("aatype_loss", aatype_loss_sum / node_num, 1, round=4)
        aatype_acc_sum = sum(log.get("aatype_acc", 0) for log in logging_outputs)
        metrics.log_scalar("aaype_acc", aatype_acc_sum/node_num, 1, round=4)

        torsion_loss_sum = sum(log.get("torsion_loss", 0) for log in logging_outputs)
        metrics.log_scalar("torsion_loss", torsion_loss_sum / node_num, 1, round=4)
        torsion_acc_sum = sum(log.get("torsion_acc", 0) for log in logging_outputs)
        metrics.log_scalar("torsion_acc", torsion_acc_sum/node_num, 1, round=4)

        # if logging_outputs[0].__contains__('traj_acc'):
        #     traj_acc_sum = sum(log.get("traj_acc", 0) for log in logging_outputs)
        #     metrics.log_scalar("traj_acc", traj_acc_sum/node_num, 1, round=4)
        #     traj_loss_sum = sum(log.get("traj_loss", 0) for log in logging_outputs)
        #     metrics.log_scalar("traj_loss", traj_loss_sum/node_num, 1, round=4)
        #     last_acc_sum = sum(log.get("last_acc", 0) for log in logging_outputs)
        #     metrics.log_scalar("last_acc", last_acc_sum/node_num, 1, round=4)
        #     last_loss_sum = sum(log.get("last_loss", 0) for log in logging_outputs)
        #     metrics.log_scalar("last_loss", last_loss_sum/node_num, 1, round=4)


    @staticmethod
    def logging_outputs_can_be_summed():
        return False
