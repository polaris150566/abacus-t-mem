from __future__ import print_function

import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

from .lig_encoder import LinearEncoder, SchNetEncoder
from .prot_encoder import ProteinMPNNEncoder
from .pifold_encoder_new import PiFoldEncoder
from .pifold_module import StructureEncoder
from .cmlm_mask import inject_noise
from .progen.modeling_progen import ProGenModel
from .progen.configuration_progen import ProGenConfig
from .save_all_atoms import write_coords
sys.path.append('/raw22/superbrain/permanent/yfliu25/ligand_protdesign/protenc_ligenc_enc_dec_pifold_all_lig_preAAenc/fasde/utils/relax')
from assess_violation import get_violation_metrics

import sys
sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
from alphafold import all_atom, r3, quat_affine
from alphafold.model2.folding import atom14_to_atom37_batch
from alphafold.data.utils.data_transforms import restype_atom37_mask, restype_atom37_to_atom14, restype_atom14_mask
from alphafold.common import residue_constants
from alphafold.common.protein import from_pdb_string

from esm import pretrained


import logging
logger = logging.getLogger(__name__)


esm_dict = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
    'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

raw_restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', 'X'
]
raw_restype_order = {restype: i for i, restype in enumerate(raw_restypes)}
raw_res_id_to_aatype = {v: k for k, v in raw_restype_order.items()}

restypes = ['<unk>', '<pad>', '<cls>', '<mask>'] + raw_restypes
restype_order = {restype: i for i, restype in enumerate(restypes)}
res_id_to_aatype = {v: k for k, v in restype_order.items()}


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

restype_1to3 = residue_constants.restype_1to3
restype_1to3.update({'X': 'UNK'})

restype_name_to_atom14_names = residue_constants.restype_name_to_atom14_names
restype_name_to_atom14_names.update({'UNK': ['N', 'CA', '', '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '']})


extend_atom_indices_from_atom14 = np.zeros((len(raw_restype_order), ))
for restype_id, restype1 in enumerate(raw_restypes):
    restype3 = restype_1to3[restype1]
    restype_atom14_names = residue_constants.restype_name_to_atom14_names[restype3]
    restype_atom14_unmask_num = sum([0 if (atom_name == '') else 1 for atom_name in restype_atom14_names])
    extend_atom_indices_from_atom14[restype_id] = restype_atom14_unmask_num-1

extend_atom_indices_from_atom14 = torch.from_numpy(extend_atom_indices_from_atom14).long()

sc_frame_mask = residue_constants.chi_angles_mask
sc_frame_mask.append([0.0, 0.0, 0.0, 0.0])
sc_frame_mask = torch.from_numpy(np.array(sc_frame_mask)).float()

extend_atom_mask = torch.tensor([0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]).float()

# C: 1, N: 2, O: 3, S: 4, None: 0, lig: 5
atom_type_id = {'C': 1, 'N': 2, 'O': 3, 'S': 4}
residue_atomtype = []
for resname, resatom in residue_constants.restype_name_to_atom14_names.items():
    res_atom_id = []
    for atomname in resatom:
        if len(atomname) > 0:
            res_atom_id.append(atom_type_id[atomname[:1]])
        else:
            res_atom_id.append(0)
    residue_atomtype.append(torch.tensor(res_atom_id))
residue_atomtype = torch.stack(residue_atomtype)
residue_atomtype = torch.cat([residue_atomtype[:20], torch.zeros(14)[None]], 0).long()

ligatom_atom14_mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def esm_refine(pred_seqs, esm_batcher, esm_model, esm_alphabet, device):
    """Use ESM-1b to refine model predicted"""

    _, _, input_ids = esm_batcher(
        [('_', seq) for seq in pred_seqs]
    )

    # input_ids = pred_ids
    results = esm_model(input_ids.to(device), repr_layers=[33], return_contacts=False)
    logits = results['logits'].detach().cpu()
    probs = F.softmax(logits, dim=-1)[..., 1:-1]
    refined_ids = logits.argmax(-1)[..., 1:-1]

    raw_cur_nll = torch.log(torch.gather(probs, -1, refined_ids[..., None])[..., 0])

    refined_seqs = [''.join([esm_alphabet.get_tok(token.item()) for token in tokens]) for tokens in refined_ids]
    refined_tokens = torch.stack([torch.tensor([restype_order[aa] for aa in seq] ).long() for seq in refined_seqs])

    return refined_seqs, refined_tokens, raw_cur_nll



# def converter_from_af2_to_esm(pred_merged_aatype, prot_mask):
#     device =pred_merged_aatype.device
#     esm_no_bos_toks = []
#     for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
#         b_esm_tokens = af2_to_esm_convert_indices[b_af2_tokens]
#         esm_no_bos_toks.append(b_esm_tokens)

#     esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)
#     esm_no_bos_toks = (1 - prot_mask) * 1 + prot_mask * esm_no_bos_toks

#     esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0)
#     esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 2)

#     return esm_toks.long()
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


MPNN_ENCODER_PRETRAIN = "/train14/superbrain/lhchen/protein_work/2023/ProteinMPNN-main/vanilla_model_weights/v_48_020.pt"
ESM_PRETRAIN = "/train14/superbrain/lhchen/protein/pretrain/esm2/params/esm2_t33_650M_UR50D.pt"

#A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design


def discrete_torsion(dist, edges=None):
    min_, max_, nbin_ = edges
    dist = (dist - min_) * nbin_ / (max_ - min_)
    dist = dist.int()
    dist = torch.clip(dist, 0, nbin_-1)
    return dist


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def flatten_prev_dims(t:torch.Tensor, no_dims:int):
    return t.reshape((-1,) + t.shape[no_dims:])


def atom14_to_atom37_batch(atom14_data,  # (B,N, 14, ...)
                     residx_atom37_to_atom14,
                     atom37_atom_exists
                    ):  # (B, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    def expand_repeat(x:torch.Tensor, n:int):
        repeat_mode= (1,)*x.dim() +(n,)
        return x.unsqueeze(-1).repeat(repeat_mode)

    atom37_data = torch.gather(atom14_data, -2, expand_repeat(residx_atom37_to_atom14, 3))
    atom37_data = atom37_data* atom37_atom_exists[..., None].type_as(atom37_data)

    return atom37_data


def make_torsion_mask_from_aatype(aatype):
    aatype = aatype - 4
    # aatype.shape [B, N]
    B, N = aatype.shape
    device = aatype.device
    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.FloatTensor(chi_angles_mask).to(aatype.device)
    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    indices = aatype.unsqueeze(-1).repeat(1, 1, 4).view(-1, 4)
    chis_mask = torch.gather(chi_angles_mask, 0, indices).view(B, N, 4)

    # bb_torsion_mask = torch.ones((B, N, 3))
    # all_torsion_mask = torch.cat([bb_torsion_mask, chis_mask], -1)

    return chis_mask


def get_violation_mask_from_design(aatype, torsion_angle, backbone_angles_sin_cos, backbone_affine_tensor):
    B, L = aatype.shape[:2]
    device = aatype.device
    aatype = torch.where(aatype-4 < 0, 20, aatype-4)

    sc_torsion_angles_sin_cos = torch.stack([torch.sin(torsion_angle), torch.cos(torsion_angle)], -1)
    aatype = aatype.reshape(B * L)
    sc_torsion_angles_sin_cos = sc_torsion_angles_sin_cos.reshape(B * L, 4, 2)
    backbone_angles_sin_cos = backbone_angles_sin_cos.reshape(B * L, 3, 2)
    torsion_angles_sin_cos = torch.cat([backbone_angles_sin_cos, sc_torsion_angles_sin_cos], 1)
    backbone_affine_tensor = backbone_affine_tensor.reshape(B * L, 4, -1)

    backb_to_global = r3.rigids_from_tensor_flat12(backbone_affine_tensor[:, 0])
    all_frames_to_global = all_atom.torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos)
    rec_atom14 = all_atom.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global )
    rec_atom14_tensor = r3.vecs_to_tensor(rec_atom14).reshape(B, L, 14, -1)

    aatype = aatype.reshape(B, L)
    residx_atom37_to_atom14 = restype_atom37_to_atom14.to(device)[aatype]
    residx_atom37_mask = restype_atom37_mask.to(device)[aatype]
    atom37 = atom14_to_atom37_batch(rec_atom14_tensor, residx_atom37_to_atom14, residx_atom37_mask)

    skeptical_mask = torch.zeros(B, L).bool()
    for write_b_idx in range(B):
        f_ctx = write_coords(atom37[write_b_idx], aatype[write_b_idx], residx_atom37_mask[write_b_idx])
        prot = from_pdb_string(f_ctx)
        struct_metrics = get_violation_metrics(prot, clash_overlap_tolerance=1.0)
        violation_idx = torch.from_numpy(struct_metrics['residue_violations']).long()
        skeptical_mask[write_b_idx, violation_idx] = True

    return skeptical_mask


def make_pdb_ctx_from_design(aatype, torsion_angle, backbone_angles_sin_cos, backbone_affine_tensor, write_b_idx=0):
    # sc_rbf_edge = self.sidechain_dist_embed(disc_torsion_angle_conti, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype)
    # tensor_flat12, rec_atom14_tensor = self.torsion_to_sidechain_frames(torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype)
    B, L = aatype.shape[:2]
    device = aatype.device
    aatype = torch.where(aatype-4 < 0, 20, aatype-4)

    sc_torsion_angles_sin_cos = torch.stack([torch.sin(torsion_angle), torch.cos(torsion_angle)], -1)
    aatype = aatype.reshape(B * L)
    sc_torsion_angles_sin_cos = sc_torsion_angles_sin_cos.reshape(B * L, 4, 2)
    backbone_angles_sin_cos = backbone_angles_sin_cos.reshape(B * L, 3, 2)
    torsion_angles_sin_cos = torch.cat([backbone_angles_sin_cos, sc_torsion_angles_sin_cos], 1)
    backbone_affine_tensor = backbone_affine_tensor.reshape(B * L, 4, -1)

    backb_to_global = r3.rigids_from_tensor_flat12(backbone_affine_tensor[:, 0])
    all_frames_to_global = all_atom.torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos)
    rec_atom14 = all_atom.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global )
    rec_atom14_tensor = r3.vecs_to_tensor(rec_atom14).reshape(B, L, 14, -1)
    tensor_flat12 = r3.rigids_to_tensor_flat12(all_frames_to_global).reshape(B, L, 8, -1)

    aatype = aatype.reshape(B, L)
    residx_atom37_to_atom14 = restype_atom37_to_atom14.to(device)[aatype]
    residx_atom37_mask = restype_atom37_mask.to(device)[aatype]
    atom37 = atom14_to_atom37_batch(rec_atom14_tensor, residx_atom37_to_atom14, residx_atom37_mask)
    f_ctx = write_coords(atom37[write_b_idx], aatype[write_b_idx], residx_atom37_mask[write_b_idx])

    return f_ctx



def get_rigid_groups(backbone_positions):
    B, L = backbone_positions.shape[:2] # B, L, 4, 3
    b = backbone_positions[:,:,1,:] - backbone_positions[:,:,0,:]
    c = backbone_positions[:,:,2,:] - backbone_positions[:,:,1,:]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + backbone_positions[:,:,1,:]
    backbone_positions = torch.cat([backbone_positions[:, :, :3], Cb[:, :, None], backbone_positions[:, :, -1][:, :, None]], 2)

    device = backbone_positions.device
    fake_aatype = torch.ones((B, L)).long().to(device) * 0
    fake_all_atom_positions = torch.cat([backbone_positions, torch.zeros((B, L, 32, 3)).float().to(device)], 2)
    fake_all_atom_mask = torch.ones_like(fake_all_atom_positions)[..., 0]

    result = all_atom.atom37_to_frames(fake_aatype, fake_all_atom_positions, fake_all_atom_mask)
    backbone_affine_tensor = result['rigidgroups_gt_frames'][:, :, :4]

    torsion_angles_result = all_atom.atom37_to_torsion_angles(fake_aatype, fake_all_atom_positions, fake_all_atom_mask)
    backbone_angles_sin_cos = torsion_angles_result['torsion_angles_sin_cos'][:, :, :3]

    return backbone_affine_tensor, backbone_angles_sin_cos



class BatchNorm(nn.Module):
    def __init__(self, num_hidden) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_hidden)

    def forward(self, input):
        # input.shape [..., H]
        input_dim = input.shape[-1]
        input_shape = input.shape
        return self.norm(input.reshape(-1, input_dim)).reshape(input_shape)




class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        # self.norm1 = BatchNorm(num_hidden)
        # self.norm2 = BatchNorm(num_hidden)
        # self.norm3 = BatchNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        # self.norm1 = BatchNorm(num_hidden)
        # self.norm2 = BatchNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class TorsionEmbedding(nn.Module):
    def __init__(self, torsion_bins, single_torsions_dim, out_dim, ops='cat',
                 torsion_noise_degree=5., embedding_scatoms=False, num_rbf=16) -> None:
        super().__init__()
        self.ops = ops
        self.torsion_bins = torsion_bins
        self.embedding_scatoms = embedding_scatoms
        self.num_rbf = num_rbf
        self.disc_to_conti = torch.linspace(-np.pi, np.pi, self.torsion_bins)
        # self.mask_scatom = mask_scatom

        self.chi1_embedder_ = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi2_embedder_ = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi3_embedder_ = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi4_embedder_ = nn.Embedding(torsion_bins, single_torsions_dim)

        self.atomtype_embedder = nn.Embedding(21, self.num_rbf)

        if (ops == 'cat'):
            self.activate_torsion = nn.Linear(single_torsions_dim * 4, out_dim)
        elif (ops == 'add'):
            self.activate_torsion = nn.Linear(single_torsions_dim, out_dim)
        else:
            raise ValueError

        noise_rad = np.deg2rad(torsion_noise_degree)
        self.vonmises_noise_dist = torch.distributions.VonMises(torch.tensor([0.]).float(), torch.tensor([1.0/noise_rad]).float())

        if embedding_scatoms:
            # self.sc_rbf_edge_act = nn.Linear(4 * 4 * self.num_rbf, out_dim, bias=False)
            # self.sc_rbf_edge_act = nn.Sequential(
            #     nn.Linear(5 * 5 * self.num_rbf, 256),
            #     nn.GELU(),
            #     nn.Linear(256, out_dim)
            # )
            self.sc_rbf_edge_act_ = nn.Sequential(
                nn.Linear(4 * 14 * 16, 256),
                nn.GELU(),
                nn.Linear(256, out_dim)
            )


    def forward(self, torsion_angle, torsion_angle_mask, E_idx=None, aatype=None,
        mask_attend=None, backbone_coords=None, mask=None, lig_mask=None, mask_scatom=True
        ):
        device = torsion_angle.device
        # chi_noise = self.vonmises_noise_dist.sample(torsion_angle.shape)[..., 0].to(device) * torsion_angle_mask
        # if self.training:
        #     torsion_angle = torsion_angle + chi_noise

        disc_torsion_angle = discrete_torsion(torsion_angle, (-np.pi, np.pi, self.torsion_bins)).long()
        # for debug: discrete_torsion(self.disc_to_conti, (-np.pi, np.pi, self.torsion_bins)).long()
        assert (len(disc_torsion_angle.shape) == 3) # B, L, 4

        chi1_embedding = self.chi1_embedder_(disc_torsion_angle[..., 0]) * torsion_angle_mask[..., 0][..., None]
        chi2_embedding = self.chi2_embedder_(disc_torsion_angle[..., 1]) * torsion_angle_mask[..., 1][..., None]
        chi3_embedding = self.chi3_embedder_(disc_torsion_angle[..., 2]) * torsion_angle_mask[..., 2][..., None]
        chi4_embedding = self.chi4_embedder_(disc_torsion_angle[..., 3]) * torsion_angle_mask[..., 3][..., None]

        if (self.ops == 'cat'):
            reduced_chi_embedding = torch.cat([chi1_embedding, chi2_embedding, chi3_embedding, chi4_embedding], -1)
        elif (self.ops == 'add'):
            reduced_chi_embedding = chi1_embedding + chi2_embedding + chi3_embedding + chi4_embedding
        act_chi_embedding = self.activate_torsion(reduced_chi_embedding)

        if self.embedding_scatoms:
            disc_torsion_angle_conti = self.disc_to_conti.to(device)[disc_torsion_angle]
            sc_rbf_edge = self.sidechain_dist_embed(disc_torsion_angle_conti, E_idx, aatype, backbone_coords, mask, lig_mask, mask_scatom) * mask_attend[..., None]
            sc_rbf_edge_act = self.sc_rbf_edge_act_(sc_rbf_edge)
            h_EC = cat_neighbors_nodes(act_chi_embedding, sc_rbf_edge_act, E_idx) * mask_attend[..., None]

            return act_chi_embedding, h_EC

        else:
            return act_chi_embedding, None


    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF


    def _get_rbf(self, A, B, E_idx, A_mask=None, B_mask=None, A_atomtype=None, B_atomtype=None):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        if A_atomtype is not None:
            A_atomtype_embedding = self.atomtype_embedder(A_atomtype.long()) #[B,L]
            B_atomtype_embedding = self.atomtype_embedder(B_atomtype.long()) #[B,L]
            A_B_atomtype_embedding = A_atomtype_embedding[:, :, None] + B_atomtype_embedding[:, None, :]
            A_B_atomtype_embedding_neighbors = gather_edges(A_B_atomtype_embedding, E_idx) #[B,L,K]
            RBF_A_B = RBF_A_B + A_B_atomtype_embedding_neighbors

        if ((A_mask is not None) and (B_mask is not None)):
            D_A_B_mask = A_mask[:, :, None] * B_mask[:, None, :] #[B, L, L]
            D_A_B_mask_attend = gather_edges(D_A_B_mask[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
            RBF_A_B = RBF_A_B * D_A_B_mask_attend[..., None]
        return RBF_A_B


    def sidechain_dist_embed(self, torsion_angle, E_idx, aatype, backbone_coords, mask, lig_mask, mask_scatom):
        B, N = aatype.shape
        device = aatype.device
        raw_aatype = aatype
        aatype = torch.where(aatype-4 < 0, 20, aatype-4)

        backbone_affine_tensor, backbone_angles_sin_cos = get_rigid_groups(backbone_coords)
        tensor_flat12, rec_atom14_tensor = self.torsion_to_sidechain_frames(
            torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype)
        sc_atom14_mask = restype_atom14_mask.to(device)[aatype]
        sc_atom14_mask = torch.where(raw_aatype[:, :, None] == 24, ligatom_atom14_mask[None, None].repeat(B, N, 1).to(device), sc_atom14_mask)
        sc_atom14_atomtype = residue_atomtype.to(device)[aatype]
        N_atom  = backbone_coords[:, :, 0, :]
        Ca_atom = backbone_coords[:, :, 1, :]
        C_atom  = backbone_coords[:, :, 2, :]
        O_atom  = backbone_coords[:, :, 3, :]


        # sc_trans = tensor_flat12[:, :, 4:, 9:]
        # sc_frame0_atom = sc_trans[:, :, 0, :]
        # sc_frame1_atom = sc_trans[:, :, 1, :]
        # sc_frame2_atom = sc_trans[:, :, 2, :]
        # sc_frame3_atom = sc_trans[:, :, 3, :]
        # # N_atom  = backbone_coords[:, :, 0, :]
        # # Ca_atom = backbone_coords[:, :, 1, :]
        # # C_atom  = backbone_coords[:, :, 2, :]
        # aatype_extend_atom_indices = extend_atom_indices_from_atom14.to(device)[aatype]
        # extent_atoms =  torch.gather(rec_atom14_tensor, -2, aatype_extend_atom_indices[..., None, None].repeat(1, 1, 1, 3))[..., 0, :]

        # if mask_scatom:
        #     sc_frame4_mask = sc_frame_mask.to(device)[aatype]
        #     sc_frame0_atom_mask = sc_frame4_mask[:, :, 0]
        #     sc_frame1_atom_mask = sc_frame4_mask[:, :, 1]
        #     sc_frame2_atom_mask = sc_frame4_mask[:, :, 2]
        #     sc_frame3_atom_mask = sc_frame4_mask[:, :, 3]
        #     sc_extend_atom_mask = extend_atom_mask.to(device)[aatype]
        #     # prot_lig_mask = (torch.any(torch.stack([(1-ligmask), protmask], -1), -1)).float()
        #     # unmask_Ca_atom_mask =  prot_lig_mask
        #     # unmask_atom_mask = mask
        # else:
        #     sc_frame0_atom_mask = None
        #     sc_frame1_atom_mask = None
        #     sc_frame2_atom_mask = None
        #     sc_frame3_atom_mask = None
        #     # unmask_Ca_atom_mask = None
        #     unmask_atom_mask = None


        # RBF_all = []
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame0_atom, E_idx, sc_frame0_atom_mask, sc_frame0_atom_mask)) #N-N
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame1_atom, E_idx, sc_frame0_atom_mask, sc_frame1_atom_mask)) #Ca-N
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame2_atom, E_idx, sc_frame0_atom_mask, sc_frame2_atom_mask)) #Ca-C
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame3_atom, E_idx, sc_frame0_atom_mask, sc_frame3_atom_mask)) #Ca-C
        # RBF_all.append(self._get_rbf(sc_frame0_atom, extent_atoms, E_idx, sc_frame0_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, Ca_atom, E_idx, sc_frame0_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, N_atom, E_idx, sc_frame0_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, C_atom, E_idx, sc_frame0_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame0_atom, E_idx, sc_frame1_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame1_atom, E_idx, sc_frame1_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame2_atom, E_idx, sc_frame1_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame3_atom, E_idx, sc_frame1_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, extent_atoms, E_idx, sc_frame1_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, Ca_atom, E_idx, sc_frame1_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, N_atom, E_idx, sc_frame1_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, C_atom, E_idx, sc_frame1_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame0_atom, E_idx, sc_frame2_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame1_atom, E_idx, sc_frame2_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame2_atom, E_idx, sc_frame2_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame3_atom, E_idx, sc_frame2_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, extent_atoms, E_idx, sc_frame2_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, Ca_atom, E_idx, sc_frame2_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, N_atom, E_idx, sc_frame2_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, C_atom, E_idx, sc_frame2_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame0_atom, E_idx, sc_frame3_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame1_atom, E_idx, sc_frame3_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame2_atom, E_idx, sc_frame3_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame3_atom, E_idx, sc_frame3_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, extent_atoms, E_idx, sc_frame3_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, Ca_atom, E_idx, sc_frame3_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, N_atom, E_idx, sc_frame3_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, C_atom, E_idx, sc_frame3_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame0_atom, E_idx, sc_extend_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame1_atom, E_idx, sc_extend_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame2_atom, E_idx, sc_extend_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame3_atom, E_idx, sc_extend_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, extent_atoms, E_idx, sc_extend_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(extent_atoms, Ca_atom, E_idx, unmask_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(extent_atoms, N_atom, E_idx, unmask_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(extent_atoms, C_atom, E_idx, unmask_atom_mask, unmask_atom_mask)) #Ca-C

        # import pdb; pdb.set_trace()
        # prot_lig_mask = (torch.any(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        # unmask_Ca_atom_mask =  prot_lig_mask
        # unmask_atom_mask = mask
        RBF_all = []
        for bb_atom_idx, backbone_atom in enumerate([N_atom, Ca_atom, C_atom, O_atom]):
            if bb_atom_idx == 0:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 2
                bb_atom_mask = mask * (1 - lig_mask)
            elif bb_atom_idx == 1:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 1
                bb_atom_mask = mask
            elif bb_atom_idx == 2:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 1
                bb_atom_mask = mask * (1 - lig_mask)
            elif bb_atom_idx == 3:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 3
                bb_atom_mask = mask * (1 - lig_mask)

            for neighbor_atom_id in range(14):
                neighbor_atom = rec_atom14_tensor[:, :, neighbor_atom_id]
                neighbor_atomtype = sc_atom14_atomtype[:, :, neighbor_atom_id]
                neighbor_atom_mask = sc_atom14_mask[:, :, neighbor_atom_id]
                RBF_all.append(self._get_rbf(backbone_atom, neighbor_atom, E_idx, bb_atom_mask, neighbor_atom_mask, bb_atomtype, neighbor_atomtype)) #N-N

        # RBF_all.append(self._get_rbf(N_atom, sc_frame0_atom, E_idx, sc_frame0_atom_mask, sc_frame0_atom_mask)) #N-N
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame1_atom, E_idx, sc_frame0_atom_mask, sc_frame1_atom_mask)) #Ca-N
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame2_atom, E_idx, sc_frame0_atom_mask, sc_frame2_atom_mask)) #Ca-C
        # RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame3_atom, E_idx, sc_frame0_atom_mask, sc_frame3_atom_mask)) #Ca-C
        # RBF_all.append(self._get_rbf(sc_frame0_atom, extent_atoms, E_idx, sc_frame0_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, Ca_atom, E_idx, sc_frame0_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, N_atom, E_idx, sc_frame0_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame0_atom, C_atom, E_idx, sc_frame0_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame0_atom, E_idx, sc_frame1_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame1_atom, E_idx, sc_frame1_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame2_atom, E_idx, sc_frame1_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame3_atom, E_idx, sc_frame1_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame1_atom, extent_atoms, E_idx, sc_frame1_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, Ca_atom, E_idx, sc_frame1_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, N_atom, E_idx, sc_frame1_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame1_atom, C_atom, E_idx, sc_frame1_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame0_atom, E_idx, sc_frame2_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame1_atom, E_idx, sc_frame2_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame2_atom, E_idx, sc_frame2_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame3_atom, E_idx, sc_frame2_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame2_atom, extent_atoms, E_idx, sc_frame2_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, Ca_atom, E_idx, sc_frame2_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, N_atom, E_idx, sc_frame2_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame2_atom, C_atom, E_idx, sc_frame2_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame0_atom, E_idx, sc_frame3_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame1_atom, E_idx, sc_frame3_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame2_atom, E_idx, sc_frame3_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame3_atom, E_idx, sc_frame3_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(sc_frame3_atom, extent_atoms, E_idx, sc_frame3_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, Ca_atom, E_idx, sc_frame3_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, N_atom, E_idx, sc_frame3_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(sc_frame3_atom, C_atom, E_idx, sc_frame3_atom_mask, unmask_atom_mask)) #Ca-C


        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame0_atom, E_idx, sc_extend_atom_mask, sc_frame0_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame1_atom, E_idx, sc_extend_atom_mask, sc_frame1_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame2_atom, E_idx, sc_extend_atom_mask, sc_frame2_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, sc_frame3_atom, E_idx, sc_extend_atom_mask, sc_frame3_atom_mask)) #C-C
        # RBF_all.append(self._get_rbf(extent_atoms, extent_atoms, E_idx, sc_extend_atom_mask, sc_extend_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(extent_atoms, Ca_atom, E_idx, unmask_atom_mask, unmask_Ca_atom_mask)) #Ca-N
        # # RBF_all.append(self._get_rbf(extent_atoms, N_atom, E_idx, unmask_atom_mask, unmask_atom_mask)) #Ca-C
        # # RBF_all.append(self._get_rbf(extent_atoms, C_atom, E_idx, unmask_atom_mask, unmask_atom_mask)) #Ca-C

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all


    def torsion_to_sidechain_frames(self, torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype):
        B, L = aatype.shape[:2]
        device = aatype.device
        # aatype = torch.where(aatype-4 < 0, 20, aatype-4)

        sc_torsion_angles_sin_cos = torch.stack([torch.sin(torsion_angle), torch.cos(torsion_angle)], -1)
        aatype = aatype.reshape(B * L)
        sc_torsion_angles_sin_cos = sc_torsion_angles_sin_cos.reshape(B * L, 4, 2)
        backbone_angles_sin_cos = backbone_angles_sin_cos.reshape(B * L, 3, 2)
        torsion_angles_sin_cos = torch.cat([backbone_angles_sin_cos, sc_torsion_angles_sin_cos], 1)
        backbone_affine_tensor = backbone_affine_tensor.reshape(B * L, 4, -1)

        backb_to_global = r3.rigids_from_tensor_flat12(backbone_affine_tensor[:, 0])
        all_frames_to_global = all_atom.torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos)
        rec_atom14 = all_atom.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global )
        rec_atom14_tensor = r3.vecs_to_tensor(rec_atom14).reshape(B, L, 14, -1)
        tensor_flat12 = r3.rigids_to_tensor_flat12(all_frames_to_global).reshape(B, L, 8, -1)
        #atom14_atom_exists = restype_atom14_mask.to(device)[aatype]

        # import pdb; pdb.set_trace()
        # # # for debug sidechain
        # aatype = aatype.reshape(B, L)
        # residx_atom37_to_atom14 = restype_atom37_to_atom14.to(device)[aatype]
        # residx_atom37_mask = restype_atom37_mask.to(device)[aatype]
        # atom37 = atom14_to_atom37_batch(rec_atom14_tensor, residx_atom37_to_atom14, residx_atom37_mask)
        # debug_f = f'/raw22/superbrain/permanent/yfliu25/ligand_protdesign/protenc_ligenc_enc_dec_pifold_all_lig_preAAenc/experiments/debug/debug_f/batch_{0}_disc10_gt_torsion0.pdb'
        # f_ctx = write_coords(atom37[0], aatype[0], residx_atom37_mask[0], debug_f)

        return tensor_flat12, rec_atom14_tensor#, atom14_atom_exists




class SideChainDecoder(nn.Module):
    def __init__(self, torsion_bins, h_VES_dim, n_embd, n_layer, n_head, dropout=0.0, torsion_noise_degree=5.) -> None:
        super().__init__()
        self.prediction_head_num = 4
        self.torsion_bins = torsion_bins

        self.activate_h_VES = nn.Linear(h_VES_dim, n_embd)

        self.sidechain_embedders_ = nn.ModuleList()
        for _ in range(self.prediction_head_num-1): # no last
            self.sidechain_embedders_.append(
                nn.Embedding(self.torsion_bins, n_embd))

        noise_rad = np.deg2rad(torsion_noise_degree)
        self.vonmises_noise_dist = torch.distributions.VonMises(torch.tensor([0.]), torch.tensor([1.0/noise_rad]))

        depth_model_args = dict(
            vocab_size=1, # fake vocab
            n_positions = self.prediction_head_num,
            n_ctx=4,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            rotary_dim=None,
            gradient_checkpointing=True,
            embd_pdrop=dropout)
        depth_config = ProGenConfig(**depth_model_args)
        self.model = ProGenModel(depth_config)

        self.prediction_heads_ = nn.ModuleList()
        for _ in range(self.prediction_head_num):
            self.prediction_heads_.append(
                nn.Linear(self.model.embed_dim, torsion_bins) )


    def forward(self, h_VES, torsion_angle, torsion_angle_mask):
        act_h_VES = self.activate_h_VES(h_VES) # B, L, D

        # chi_noise = self.vonmises_noise_dist(torsion_angle.shape)[..., 0]
        # if self.training:
        #     torsion_angle = torsion_angle + chi_noise
        disc_torsion_angle = discrete_torsion(torsion_angle, (-np.pi, np.pi, self.torsion_bins))

        bsz, res_num = disc_torsion_angle.shape[:2] # B, L, 4
        inputs_embeds = [act_h_VES[:, :, None]] # B, L, 1, D

        for sc_idx, sc_embedder in enumerate(self.sidechain_embedders_):
            embeds_sum =  sc_embedder(disc_torsion_angle[..., sc_idx]) * torsion_angle_mask[..., sc_idx][..., None]
            inputs_embeds.append(embeds_sum[:, :, None]) # B, L, 1, D

        inputs_embeds = torch.cat(inputs_embeds, 2) # B, L, 4, D
        inputs_embeds = inputs_embeds.reshape(bsz*res_num, 4,-1) # BxL, 4, D
        if self.training:
            use_gradient_checkpoint= True
        else:
            use_gradient_checkpoint = False
        dec_output = self.model(inputs_embeds = inputs_embeds, input_bias = 0.0, input_bias_layer = -1, use_gradient_checkpoint = use_gradient_checkpoint)
        dec_output = dec_output[0].reshape(bsz, res_num, 4, -1) # B, L, 4, D

        pred_logits = []
        for pred_idx, pred_head in enumerate(self.prediction_heads_):
            pred_logits.append(pred_head(dec_output[..., pred_idx, :])[..., None, :])

        pred_logits = torch.cat(pred_logits, 2)  # B, L, 4, D

        return pred_logits

    @ torch.no_grad()
    def sample(self, h_VES, torsion_angle_mask, temperature=1.0):
        # forward the model to get the logits for the index in the sequence
        act_h_VES = self.activate_h_VES(h_VES) # B, L, D
        bsz, res_num = act_h_VES.shape[:-1]
        raw_inputs_embeds = [act_h_VES[:, :, None]]
        last_sc_disc_torsion_angle = None
        stacked_sc_disc_torsion_angle = []
        stacked_sc_disc_torsion_angle_prob = []
        for sc_idx in range(4):
            if (sc_idx > 0):
                cur_sc_embeds = self.sidechain_embedders_[sc_idx-1](last_sc_disc_torsion_angle) * torsion_angle_mask[..., sc_idx-1][..., None] # B, L, D
                raw_inputs_embeds.append(cur_sc_embeds[:, :, None])

            inputs_embeds = torch.cat(raw_inputs_embeds, 2) # B, L, X, D
            inputs_embeds = inputs_embeds.reshape(bsz*res_num, sc_idx+1, -1) # BxL, X, D

            if self.training:
                use_gradient_checkpoint = True
            else:
                use_gradient_checkpoint = False

            dec_output = self.model(inputs_embeds = inputs_embeds,  input_bias = 0.0, input_bias_layer = -1, use_gradient_checkpoint = use_gradient_checkpoint)
            dec_output = dec_output[0][:, -1].reshape(bsz*res_num, -1) # BxL, D

            cur_pred_torsion_angle_logits = self.prediction_heads_[sc_idx](dec_output) # BxL, D
            logits = cur_pred_torsion_angle_logits / temperature
            probs = F.softmax(logits, dim=-1)  # B, L, D
            last_sc_disc_torsion_angle = torch.multinomial(probs, num_samples=1).reshape(bsz, res_num) # B, L
            stacked_sc_disc_torsion_angle.append(last_sc_disc_torsion_angle[..., None])

            raw_cur_ll = torch.log(torch.gather(probs.reshape(bsz, res_num, -1), -1, last_sc_disc_torsion_angle[..., None])[..., 0])
            stacked_sc_disc_torsion_angle_prob.append(raw_cur_ll)

        return torch.cat(stacked_sc_disc_torsion_angle, -1), torch.stack(stacked_sc_disc_torsion_angle_prob, -1)


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, num_chain_embeddings=16, embed_prot_lig_edge=False):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        # self.embed_sc_torsion = embed_sc_torsion

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

        if embed_prot_lig_edge:
            self.edge_type_embedding = nn.Embedding(4, edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels, lig_mask=None):
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        if (lig_mask is not None):
            RBF_all[lig_mask.bool(),:,16:]=0
        offset = residue_idx[:,:,None]-residue_idx[:,None,:]  #记录的是氨基酸序列上的差距
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)

        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        edge_type = (1 - mask)[:,:,None] * (1 - mask)[:,None, :] * 3
        if (lig_mask is not None):
            edge_type = (lig_mask[:,:,None] + lig_mask[:,None,:]) + edge_type
        E_edge_type = gather_edges(edge_type[:,:,:,None], E_idx)[:,:,:,0]
        E_type_embed = self.edge_type_embedding(E_edge_type.long()) # 0: protprot; 1: prot-lig; 2: lig-lig; 3: mask
        E = E + E_type_embed
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=5,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1,
        lig_embed_mode='schnet', lig_h_dim=200, lig_E_dim=5, lig_hidden=128,
        pre_prot_embed=True, pre_prot_mode='proteinMPNN',
        nar=False, noise='random_mask', lig_neighbor_seq_mask=True,
        mpnn_encoder_pretrain=MPNN_ENCODER_PRETRAIN, freeze_encoder_param=True,
        ligmpnn_init=False,
        esm_embedder=False,
        max_iter_num=4,
        embed_unimol_reprs=False,
        torsion_angle_gap=5,
        tosion_decoder_layer=3,
        tosion_decoder_head=4,
        torsion_loss_weight=0.6,
        avail_torsion_loss_weight=0.15,
        encode_mpnn=False,
        aux_enc_sc='PiFold',
        ):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.pre_prot_embed = pre_prot_embed
        self.pre_prot_mode = pre_prot_mode
        self.nar = nar
        self.noise = noise
        self.lig_neighbor_seq_mask = lig_neighbor_seq_mask
        self.ligmpnn_init = ligmpnn_init
        self.esm_embedder = esm_embedder
        self.max_iter_num = max_iter_num
        self.embed_unimol_reprs = embed_unimol_reprs
        self.encode_mpnn = encode_mpnn
        self.torsion_bins = 360//torsion_angle_gap
        self.torsion_loss_weight = torsion_loss_weight
        self.avail_torsion_loss_weight = avail_torsion_loss_weight
        self.augment_eps = augment_eps
        self.aux_enc_sc = aux_enc_sc

        if pre_prot_embed:
            if encode_mpnn:
                self.prot_encoder_mpnn = ProteinMPNNEncoder(
                    node_features, edge_features, hidden_dim, k_neighbors=48, augment_eps=0.2)
            self.prot_encoder_pifold = PiFoldEncoder()

        self.W_esm_LN = nn.LayerNorm(1280)
        self.W_esm_act = nn.Linear(1280, hidden_dim)


        if (lig_embed_mode == 'linear'):
            self.lig_encoder = LinearEncoder(lig_h_dim, lig_hidden)
        elif (lig_embed_mode == 'schnet'):
            self.lig_encoder = SchNetEncoder(lig_h_dim, lig_E_dim, hidden_channels=lig_hidden)
        else:
            raise NotImplementedError
        if self.embed_unimol_reprs:
            self.unimol_repr_activate = nn.Sequential(
                nn.Linear(512, lig_hidden),
                nn.ReLU(),
                nn.Linear(lig_hidden, lig_hidden))

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, embed_prot_lig_edge=True)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.lig_plip_itype_embedding = nn.Embedding(14, hidden_dim)
        self.W_c = TorsionEmbedding(self.torsion_bins, hidden_dim, hidden_dim, embedding_scatoms=True)

        if (aux_enc_sc == 'PiFold'):
            self.aux_scEnc = StructureEncoder(hidden_dim, num_encoder_layers=4, dropout=0.1)
        elif (aux_enc_sc == 'MPNN'):
            self.aux_scEnc = nn.ModuleList([EncLayer(hidden_dim, 2 * hidden_dim, dropout=0.1) for _ in range(4)])
        self.activate_W_sc_node = nn.Linear(2 * hidden_dim, hidden_dim)
        self.activate_W_sc_edge = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.activate_W_sc_merge_edge = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

        self.h_V_segment = nn.Embedding(4, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        # self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_letters, bias=True)
        )

        self.sc_decoder = SideChainDecoder(
            self.torsion_bins, hidden_dim * 2, hidden_dim, tosion_decoder_layer, tosion_decoder_head, dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if (encode_mpnn and pre_prot_embed):
            self._initialize_pretrained_protein_encoder(mpnn_encoder_pretrain, freeze_param=freeze_encoder_param)

        if ligmpnn_init:
            self._initialize_ligmpnn_encoder_decoder(mpnn_encoder_pretrain)

        # if esm_embedder:
        #     self._freeze_esm_parameter()


    def _initialize_pretrained_protein_encoder(self, mpnn_encoder_pretrain, freeze_param=False):
        if (self.pre_prot_mode == 'proteinMPNN'):
            ckpt = torch.load(mpnn_encoder_pretrain, map_location='cpu')
            state_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in self.prot_encoder_mpnn.state_dict()}
            self.prot_encoder_mpnn.load_state_dict(state_dict)
            logger.info(f'protein encoder init from {mpnn_encoder_pretrain}')

        if freeze_param:
            for name, param in self.prot_encoder_mpnn.named_parameters():
                param.requires_grad = False
            logger.info(f'freeze parameters of protein encoder')


    def _initialize_ligmpnn_encoder_decoder(self, mpnn_encoder_pretrain, encoder=True, decoder=True):
        raise NotImplementedError
        # ckpt = torch.load(mpnn_encoder_pretrain, map_location='cpu')
        # state_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in self.prot_encoder.state_dict()}
        # self.prot_encoder.load_state_dict(state_dict)
        # logger.info(f'protein encoder init from {mpnn_encoder_pretrain}')


    # def _freeze_esm_parameter(self, ):
    #     for name, param in self.esm_model.named_parameters():
    #         param.requires_grad = False
    #     logger.info(f'freeze parameters of esm model')


    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        lig_node_attr, lig_edge_attr, lig_edge_index,
        use_input_decoding_order=False, decoding_order=None,
        lig_mask=None, seq_mask=None,
        esm_embedding=None, last_iter_S_embed=None, unimol_reprs=None,
        torsion_angles=None, alt_chi_angles=None, torsion_angle_mask=None,
        prev_tokens=None,
        plip_anno_itype_list=None
        ):
        """ Graph-conditioned sequence model """

        device=X.device
        B, L = X.shape[:2]
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        chain_M = chain_M*mask #update chain_M to include missing regions
        torsion_angle_mask = torsion_angle_mask * prot_mask[..., None]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        if self.pre_prot_embed:
            # Prepare node and edge embeddings
            prot_X = X * prot_mask[..., None, None]
            pre_h_V_encoder = self.prot_encoder_pifold(prot_X, prot_mask, S)

            if self.encode_mpnn:
                pre_h_V_encoder_mpnn = self.prot_encoder_mpnn(prot_X, S, prot_mask, residue_idx, chain_encoding_all)
                pre_h_V_encoder = pre_h_V_encoder_mpnn + pre_h_V_encoder
        else:
            pre_h_V_encoder = 0

        all_zeros_lig_mask = torch.all(lig_mask == 0, 1)
        all_zeros_lig_unmask_bidx = torch.arange(B).long().to(device)[~all_zeros_lig_mask]
        lig_node = torch.zeros((B, L, self.hidden_dim), device=device)
        lig_X = X[:, :, 1] * lig_mask[..., None]

        if (len(all_zeros_lig_unmask_bidx) > 0):
            unmasked_lig_X = lig_X[all_zeros_lig_unmask_bidx]
            unmasked_lig_node_attr = lig_node_attr[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_index = lig_edge_index[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_attr = lig_edge_attr[all_zeros_lig_unmask_bidx]
            unmasked_unimol_reprs = unimol_reprs[all_zeros_lig_unmask_bidx]

            unmasked_lig_node = self.lig_encoder(unmasked_lig_node_attr, unmasked_lig_X, unmasked_lig_edge_index, unmasked_lig_edge_attr)
            if self.embed_unimol_reprs:
                unmasked_unimol_reprs = self.unimol_repr_activate(unmasked_unimol_reprs)
                unmasked_lig_node = (unmasked_lig_node + unmasked_unimol_reprs) * lig_mask[all_zeros_lig_unmask_bidx][..., None]
            lig_node[all_zeros_lig_unmask_bidx] = unmasked_lig_node

        lig_node = lig_node + self.lig_plip_itype_embedding(plip_anno_itype_list.long())
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None] * lig_node + prot_mask[..., None] * pre_h_V_encoder

        seg_token = prot_mask * 1 + lig_mask * 2
        h_V_seg = self.h_V_segment(seg_token.long())
        h_V = h_V + h_V_seg
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]

        if not self.nar:
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

            if self.lig_neighbor_seq_mask:
                assert (seq_mask is not None)
                unmasked_seq = 1 - seq_mask.float()
                order_mask_backward = order_mask_backward * unmasked_seq[:, :, None]

            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend # mask_attend 1: seq involved
            mask_fw = mask_1D * (1. - mask_attend)

            prev_tokens = S
            prev_torsion=torsion_angles
            prev_torsion_mask = torsion_angle_mask


        else:
            #### only mask some of seq_mask
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
            # mask_bw = mask_1D * mask_attend
            mask_bw = mask_1D * (1. - mask_attend)
            mask_fw = mask_1D * (1. - mask_attend)

            # if self.lig_neighbor_seq_mask:
            #     avail_unmask = seq_mask.float() * prot_mask
            # else:
            #     avail_unmask = prot_mask
            # ## DEBUG: prev_tokens = (S * (1 - avail_unmask)).long()
            # prev_tokens = inject_noise(S, avail_unmask.bool(), noise=self.noise)

            sc_mask = (torch.all(torch.stack([(prev_tokens >= 4), prev_tokens!=24], -1), -1)).float()
            prev_torsion = sc_mask[..., None] * torsion_angles
            prev_alt_torsion = sc_mask[..., None] * torsion_angles
            prev_torsion_mask = sc_mask[..., None] * torsion_angle_mask

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(prev_tokens) + self.W_esm_act(self.W_esm_LN(last_iter_S_embed))

        # sc_mask_attend = gather_nodes(sc_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        sc_mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        sc_mask_attend = mask.unsqueeze(-1) * sc_mask_attend
        # for debug: visual atom14 in pymol
        # h_C, h_EC_ = self.W_c(torsion_angles, torsion_angle_mask, E_idx, aatype=S, mask_attend=sc_mask_attend, backbone_coords=X, ligmask=lig_mask, protmask=prot_mask)

        h_C, h_EC_ = self.W_c(prev_torsion, prev_torsion_mask, E_idx, aatype=prev_tokens,
            mask_attend=sc_mask_attend, backbone_coords=X, mask=mask, lig_mask=lig_mask, mask_scatom=True) # hidden_dim, 2 * hidden_dim
        h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1)) #B,L,D
        h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES #[B, L, 48, 256]

        if (len(self.aux_enc_sc) > 0):
            h_ESC = self.activate_W_sc_merge_edge(h_ESC)#一个线性连接层
            scEnc_h_V = h_SC + h_V
            scEnc_h_E = h_ESC + h_E
            # scEnc_h_V = self.activate_scEnc_h_V(torch.cat([h_SC, h_V], -1))
            # scEnc_h_E = self.activate_scEnc_h_E(cat_neighbors_nodes(scEnc_h_V, torch.cat([h_ESC, h_E], -1), E_idx))
            if (self.aux_enc_sc == 'PiFold'):
                mask_pair = (mask[:, :, None] * mask[:,None,:])
                edge_mask = gather_edges(mask_pair[:,:,:,None], E_idx)[:,:,:,0].bool()# B,L,K ，Eidx就是B,L,K
                h_SC_update, h_ESC_update = self.aux_scEnc(scEnc_h_V, scEnc_h_E, E_idx, mask, edge_mask) # hidden_dim

            elif (self.aux_enc_sc == 'MPNN'):
                mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
                mask_attend = mask.unsqueeze(-1) * mask_attend
                for layer in self.aux_scEnc:
                    scEnc_h_V, scEnc_h_E = layer(scEnc_h_V, scEnc_h_E, E_idx, mask, mask_attend)#  B,L,D |B,L,K,D
                h_SC_update = scEnc_h_V #B,L,D
                h_ESC_update = scEnc_h_E # B,L,K,D

            # h_V = torch.cat([h_SC_update, h_V], -1)
            ############################################
            h_V = h_SC_update + h_V
            ############################################
            h_ESC = torch.cat([h_E, h_ESC_update], -1)
            h_EX_encoder = cat_neighbors_nodes(h_SC_update, h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) # 2 * hidden

        else:
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) # 2 * hidden

        # Build encoder embeddings
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx) # 2 * hidden
            h_ESV = mask_bw * (h_ESCV) + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        ## teacher forcing training p(sc^i | aa^i, aa^j, sc^j) * p(aa^i | aa^j, sc^j) = p(aa^i, sc^i | aa^j, sc^j)
        all_h_S = self.W_s(S)
        sc_res_ctx = torch.cat([all_h_S, h_V], -1)
        sc_logits = self.sc_decoder(sc_res_ctx, torsion_angles, torsion_angle_mask)
        log_sc_probs = F.log_softmax(sc_logits, dim=-1)

        # # Sample from Gumbel-Softmax distribution
        # sc_torsion_disc = F.gumbel_softmax(sc_logits, 1.0, hard=True)
        # pred_sc_torsion = torch.sum(torch.linspace(-np.pi, np.pi, 360//5).to(device).view(*([1] * (len(sc_logits.shape) - 1)), -1) * sc_torsion_disc, dim=-1)


        return log_probs, prev_tokens, log_sc_probs, prev_torsion_mask


    def nar_sample(self,
                   X,
                   S,
                   mask,
                   chain_M,
                   residue_idx,
                   chain_encoding_all,
        lig_node_attr, lig_edge_attr, lig_edge_index,
        lig_mask, iter_num=5, temperature=1.0, seq_mask_pocket=None,
        backbone_affine_tensor=None, backbone_angles_sin_cos=None,
        esm_embedding=None,
        unimol_reprs=None,
        initial_S=None,
        esm_batcher=None, esm_model=None, esm_alphabet=None, mask_mode='aatype_nll',
        return_pdb_ctx_iter=-1,
        packing_only=False, noise_scale=1.0,
        start_from_init=False,
        bg_dist=None,
        bg_weight=0.5,
        fixed_positions_mask=None,
        plip_anno_itype_list=None
        ):

        device=X.device
        B, L = X.shape[:2]
        esm_embedding = None
        res_num = X.shape[-1]
        bsz = X.shape[0]
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        prot_len = prot_mask.sum(-1)[0].long().item()
        chain_M = chain_M*mask #update chain_M to include missing regions
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X) * noise_scale

        initial_S_ = S.clone().float()
        initial_S_pocket = initial_S_ * (1 - seq_mask_pocket.float()) + lig_mask * 24 + (1 - mask)

        if initial_S is None:
            initial_S = initial_S_ * (1 - prot_mask)
        #     given_initial_S = False
        # else:
        #     given_initial_S = True

        if isinstance(return_pdb_ctx_iter, int):
            return_pdb_ctx_iter = [return_pdb_ctx_iter]
        import pdb; pdb.set_trace()
        if self.pre_prot_embed:
            # Prepare node and edge embeddings
            prot_X = X * prot_mask[..., None, None]
            pre_h_V_encoder = self.prot_encoder_pifold(prot_X, prot_mask, S)
            if self.encode_mpnn:
                pre_h_V_encoder_mpnn = self.prot_encoder_mpnn(prot_X, S, prot_mask, residue_idx, chain_encoding_all)
                pre_h_V_encoder = pre_h_V_encoder_mpnn + pre_h_V_encoder
        else:
            pre_h_V_encoder = 0

        all_zeros_lig_mask = torch.all(lig_mask == 0, 1)
        all_zeros_lig_unmask_bidx = torch.arange(B).long().to(device)[~all_zeros_lig_mask]
        lig_node = torch.zeros((B, L, self.hidden_dim), device=device)
        lig_X = X[:, :, 1] * lig_mask[..., None]

        if (len(all_zeros_lig_unmask_bidx) > 0):
            unmasked_lig_X = lig_X[all_zeros_lig_unmask_bidx]
            unmasked_lig_node_attr = lig_node_attr[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_index = lig_edge_index[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_attr = lig_edge_attr[all_zeros_lig_unmask_bidx]
            unmasked_unimol_reprs = unimol_reprs[all_zeros_lig_unmask_bidx]

            unmasked_lig_node = self.lig_encoder(unmasked_lig_node_attr, unmasked_lig_X, unmasked_lig_edge_index, unmasked_lig_edge_attr)
            if self.embed_unimol_reprs:
                unmasked_unimol_reprs = self.unimol_repr_activate(unmasked_unimol_reprs)
                unmasked_lig_node = (unmasked_lig_node + unmasked_unimol_reprs) * lig_mask[all_zeros_lig_unmask_bidx][..., None]
            lig_node[all_zeros_lig_unmask_bidx] = unmasked_lig_node

        lig_node = lig_node + self.lig_plip_itype_embedding(plip_anno_itype_list.long())
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None]*lig_node + prot_mask[..., None] * pre_h_V_encoder
        seg_token = prot_mask * 1 + lig_mask * 2
        h_V_seg = self.h_V_segment(seg_token.long())
        h_V = h_V + h_V_seg
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_init = h_V
        h_E_init = h_E
        cur_iter_masked_S = initial_S
        # if not given_initial_S:
        init_iter_mask = cur_iter_masked_S == 0
        # else:
        #     init_iter_mask = prot_mask == 1
        init_iter_mask_pocket = initial_S_pocket == 0

        cur_iter_S = None
        iter_seq_list = []
        iter_identity_list = []
        iter_identity_list_pocket = []
        iter_torsion_list = []
        iter_torsion_discrete_list = []
        # given pre-designed torsion information in cur_iter_torsion
        cur_iter_torsion = torch.zeros((B, L, 4)).float().to(device)
        cur_iter_torsion_mask = torch.zeros((B, L, 4)).float().to(device)
        # if side-chain packing only, then cur_iter_torsion is None
        for cur_iter_num in range(iter_num):
            # import pdb; pdb.set_trace()
            cur_iter_mask = cur_iter_masked_S == 0

            if (cur_iter_num > 0):
                input_esm_tokens, raw_esm_tokens = converter_from_af2_to_esm(cur_iter_S, prot_mask)
                with torch.no_grad():
                    last_iter_esm_embed = torch.zeros((B, L, 1280)).to(device)
                    output = esm_model(input_esm_tokens, repr_layers=[33], return_contacts=False)
                    esm_rep_ = output['representations'][33][:, 1:-1].detach()
                    last_iter_S_embed = torch.where((prot_mask == 1)[..., None], esm_rep_, last_iter_esm_embed)
            else:
                last_iter_S_embed = torch.zeros((B, L, 1280)).to(device)

            # Concatenate sequence embeddings for autoregressive decoder
            h_S = self.W_s(cur_iter_masked_S.long()) + self.W_esm_act(self.W_esm_LN(last_iter_S_embed))

            sc_mask = (torch.all(torch.stack([(cur_iter_masked_S >= 4), cur_iter_masked_S!=24], -1), -1)).float()
            cur_iter_torsion = sc_mask[..., None] * cur_iter_torsion
            cur_iter_torsion_mask = sc_mask[..., None] * cur_iter_torsion_mask
            sc_mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
            sc_mask_attend = mask.unsqueeze(-1) * sc_mask_attend

            h_C, h_EC_ = self.W_c(cur_iter_torsion, cur_iter_torsion_mask, E_idx, aatype=cur_iter_masked_S.long(),
                mask_attend=sc_mask_attend, backbone_coords=X, mask=mask, lig_mask=lig_mask, mask_scatom=True)
            h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1))
            h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES #[B, L, 48, 256]

            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
            # mask_bw = mask_1D * mask_attend
            mask_bw = mask_1D * (1. - mask_attend)
            mask_fw = mask_1D * (1. - mask_attend)

            # Build encoder embeddings
            if (len(self.aux_enc_sc) > 0):
                h_ESC = self.activate_W_sc_merge_edge(h_ESC)#一个线性连接层
                scEnc_h_V = h_SC + h_V
                scEnc_h_E = h_ESC + h_E
                # scEnc_h_V = self.activate_scEnc_h_V(torch.cat([h_SC, h_V], -1))
                # scEnc_h_E = self.activate_scEnc_h_E(cat_neighbors_nodes(scEnc_h_V, torch.cat([h_ESC, h_E], -1), E_idx))
                if (self.aux_enc_sc == 'PiFold'):
                    mask_pair = (mask[:, :, None] * mask[:,None,:])
                    edge_mask = gather_edges(mask_pair[:,:,:,None], E_idx)[:,:,:,0].bool()
                    h_SC_update, h_ESC_update = self.aux_scEnc(scEnc_h_V, scEnc_h_E, E_idx, mask, edge_mask) # hidden_dim

                elif (self.aux_enc_sc == 'MPNN'):
                    mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
                    mask_attend = mask.unsqueeze(-1) * mask_attend
                    for layer in self.aux_scEnc:
                        scEnc_h_V, scEnc_h_E = layer(scEnc_h_V, scEnc_h_E, E_idx, mask, mask_attend)#  B,L,D |B,L,K,D
                    h_SC_update = scEnc_h_V #B,L,D
                    h_ESC_update = scEnc_h_E # B,L,K,D

                # h_V = torch.cat([h_SC_update, h_V], -1)
                ############################################
                h_V = h_SC_update + h_V
                ############################################
                h_ESC = torch.cat([h_E, h_ESC_update], -1)
                h_EX_encoder = cat_neighbors_nodes(h_SC_update, h_E, E_idx)
                h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) # 2 * hidden

            else:
                h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
                h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx)
                h_ESV = mask_bw * h_ESCV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            probs = F.softmax(logits, dim=-1)
            if bg_dist is not None and bg_weight > 0:
                probs = (1 - bg_weight) * probs + bg_weight * bg_dist
                logits = torch.log(probs + 1e-8)
            raw_S_iter, raw_S_iter_score = sample_from_categorical(logits, temperature)

            cur_iter_S = (cur_iter_mask.float() * raw_S_iter + cur_iter_masked_S).long()
            cur_iter_S = cur_iter_S.masked_fill((1-prot_mask).bool(), 24.0)

            # if (esm_batcher is not None):
            #     seq_list = []
            #     for b_aatype in (cur_iter_S * prot_mask).cpu().numpy():
            #         sequence = ''.join([res_id_to_aatype[res_id] for res_id in b_aatype if (res_id not in [0, 1, 2, 3])])
            #         seq_list.append(sequence)

            #     prot_aa_num = len(seq_list[0])
            #     refined_seqs, refined_tokens, refined_raw_cur_nll = esm_refine(seq_list, esm_batcher, esm_model, esm_alphabet, device)

            #     cur_iter_S[:, :prot_aa_num] = refined_tokens
            #     # import pdb; pdb.set_trace()
            #     output_scores = (1 - init_iter_mask.float()) * 2# + refined_raw_cur_nll * init_iter_mask.float()
            #     output_scores[:, :prot_aa_num] = refined_raw_cur_nll
            # else:
            raw_cur_nll = torch.log(torch.gather(probs, -1, cur_iter_S[..., None])[..., 0])
            output_scores = (1 - init_iter_mask.float()) * 2 + raw_cur_nll * init_iter_mask.float()

            if start_from_init:
                if (cur_iter_num == 0):
                    cur_iter_S = S.long()

            if fixed_positions_mask is not None:
                cur_iter_S = torch.where(fixed_positions_mask.bool(), S.long(), cur_iter_S.long())

            if packing_only:
                cur_iter_S = S.long()

            all_h_S = self.W_s(cur_iter_S)
            sc_res_ctx = torch.cat([all_h_S, h_V], -1)
            # import pdb; pdb.set_trace()

            cur_iter_torsion_mask = make_torsion_mask_from_aatype(cur_iter_S) # cur_iter_S
            cur_iter_disc_torsion, cur_iter_disc_torsion_ll = self.sc_decoder.sample(sc_res_ctx, cur_iter_torsion_mask, temperature)
            cur_iter_torsion = self.W_c.disc_to_conti.to(device)[cur_iter_disc_torsion]

            iter_seq_list.append(cur_iter_S)
            iter_torsion_list.append(cur_iter_torsion)
            iter_torsion_discrete_list.append(cur_iter_disc_torsion)

            cur_batch_iter_ident = ((cur_iter_S == S).float() * init_iter_mask.float()).sum(-1)/(init_iter_mask.float() + 1e-10).sum(-1)
            iter_identity_list.append(cur_batch_iter_ident) # all identity
            cur_batch_iter_ident_pocket = ((cur_iter_S == S).float() * init_iter_mask_pocket.float()).sum(-1)/(init_iter_mask_pocket.float() + 1e-10).sum(-1)
            iter_identity_list_pocket.append(cur_batch_iter_ident_pocket) # pocket identity

            if (mask_mode == 'violation'):
                skeptical_mask_prot = get_violation_mask_from_design(cur_iter_S[:, :prot_len], cur_iter_torsion[:, :prot_len], backbone_angles_sin_cos[:, :prot_len], backbone_affine_tensor[:, :prot_len])
                cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot.to(device), 0.0)
                cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)

            else:
                output_scores_mask = init_iter_mask
                cur_iter_p = 1 - (cur_iter_num + 1) / iter_num
                iter_output_scores = output_scores
                iter_output_scores = iter_output_scores * (cur_iter_masked_S == 0).float() + 2 * (cur_iter_masked_S != 0).float()
                skeptical_mask_prot = _skeptical_unmasking(iter_output_scores[:,:prot_len], output_scores_mask[:, :prot_len], cur_iter_p)
                cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot, 0.0)
                cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)

            h_V = h_V_init
            h_E = h_E_init

        iter_seq_list = torch.stack(iter_seq_list)
        iter_identity_list = torch.stack(iter_identity_list)
        iter_identity_list_pocket = torch.stack(iter_identity_list_pocket)

        iter_torsion_discrete_list = torch.stack(iter_torsion_discrete_list)
        iter_torsion_list = torch.stack(iter_torsion_list)

        design_ctx = {
            'aatype': iter_seq_list.cpu(),
            'all_identity': iter_identity_list.cpu(),
            'pocket_identity': iter_identity_list_pocket.cpu(),
            'protein_mask': prot_mask,
            'torsion_discrete': iter_torsion_discrete_list,
            'torsion_continuous': iter_torsion_list
        }

        return design_ctx
