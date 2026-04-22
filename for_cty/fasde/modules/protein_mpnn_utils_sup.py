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
from .cmlm_mask import inject_noise
from .progen.modeling_progen import ProGenModel
from .progen.configuration_progen import ProGenConfig
from .save_all_atoms import write_coords

import sys
sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
from alphafold import all_atom, r3, quat_affine
from alphafold.model2.folding import atom14_to_atom37_batch
from alphafold.data.utils.data_transforms import restype_atom37_mask, restype_atom37_to_atom14, restype_atom14_mask
from alphafold.common import residue_constants

from esm import pretrained


import logging
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



def converter_from_af2_to_esm(pred_merged_aatype, rec_len, prot_mask):
    device =pred_merged_aatype.device
    esm_no_bos_toks = []
    for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
        b_esm_tokens = af2_to_esm_convert_indices[b_af2_tokens]
        esm_no_bos_toks.append(b_esm_tokens)

    esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)
    esm_no_bos_toks = (1 - prot_mask) + prot_mask * esm_no_bos_toks

    esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0)
    esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 1)
    for b_idx, b_prot_len in enumerate(rec_len):
        esm_toks[b_idx][b_prot_len.item()+1] = 2

    return esm_toks.long()


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
    def __init__(self, torsion_bins, single_torsions_dim, out_dim, ops='cat', torsion_noise_degree=5., embedding_scatoms=False, num_rbf=16) -> None:
        super().__init__()
        self.ops = ops
        self.torsion_bins = torsion_bins
        self.embedding_scatoms = embedding_scatoms
        self.num_rbf = num_rbf
        self.disc_to_conti = torch.linspace(-np.pi, np.pi, self.torsion_bins)

        self.chi1_embedder = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi2_embedder = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi3_embedder = nn.Embedding(torsion_bins, single_torsions_dim)
        self.chi4_embedder = nn.Embedding(torsion_bins, single_torsions_dim)

        if (ops == 'cat'):
            self.activate_torsion = nn.Linear(single_torsions_dim * 4, out_dim)
        elif (ops == 'add'):
            self.activate_torsion = nn.Linear(single_torsions_dim, out_dim)
        else:
            raise ValueError

        noise_rad = np.deg2rad(torsion_noise_degree)
        self.vonmises_noise_dist = torch.distributions.VonMises(torch.tensor([0.]).float(), torch.tensor([1.0/noise_rad]).float())

        if embedding_scatoms:
            self.sc_rbf_edge_act = nn.Sequential(
                nn.Linear(4 * 4 * self.num_rbf, 256),
                nn.GELU(),
                nn.Linear(256, out_dim)
            )
            

    def forward(self, torsion_angle, torsion_angle_mask, E_idx=None, backbone_affine_tensor=None, backbone_angles_sin_cos=None, aatype=None, mask_attend=None):
        device = torsion_angle.device
        # chi_noise = self.vonmises_noise_dist.sample(torsion_angle.shape)[..., 0].to(device) * torsion_angle_mask
        # if self.training:
        #     torsion_angle = torsion_angle + chi_noise

        disc_torsion_angle = discrete_torsion(torsion_angle, (-np.pi, np.pi, self.torsion_bins)).long()
        # for debug: discrete_torsion(self.disc_to_conti, (-np.pi, np.pi, self.torsion_bins)).long()
        assert (len(disc_torsion_angle.shape) == 3) # B, L, 4

        chi1_embedding = self.chi1_embedder(disc_torsion_angle[..., 0]) * torsion_angle_mask[..., 0][..., None]
        chi2_embedding = self.chi2_embedder(disc_torsion_angle[..., 1]) * torsion_angle_mask[..., 1][..., None]
        chi3_embedding = self.chi3_embedder(disc_torsion_angle[..., 2]) * torsion_angle_mask[..., 2][..., None]
        chi4_embedding = self.chi4_embedder(disc_torsion_angle[..., 3]) * torsion_angle_mask[..., 3][..., None]

        if (self.ops == 'cat'):
            reduced_chi_embedding = torch.cat([chi1_embedding, chi2_embedding, chi3_embedding, chi4_embedding], -1)
        elif (self.ops == 'add'):
            reduced_chi_embedding = chi1_embedding + chi2_embedding + chi3_embedding + chi4_embedding
        act_chi_embedding = self.activate_torsion(reduced_chi_embedding)
        
        if self.embedding_scatoms:
            disc_torsion_angle_conti = self.disc_to_conti.to(device)[disc_torsion_angle]
            # sc_rbf_edge = self.sidechain_dist_embed(disc_torsion_angle_conti, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype)
            sc_rbf_edge = self.sidechain_dist_embed(disc_torsion_angle_conti, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype)
            sc_rbf_edge_ = sc_rbf_edge * mask_attend[..., None]
            sc_rbf_edge_act = self.sc_rbf_edge_act(sc_rbf_edge_)
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


    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


    def sidechain_dist_embed(self, torsion_angle, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype):
        tensor_flat12, rec_atom14_tensor = self.torsion_to_sidechain_frames(torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype)

        sc_trans = tensor_flat12[:, :, 4:, 9:]
        sc_frame0_atom = sc_trans[:, :, 0, :]
        sc_frame1_atom = sc_trans[:, :, 1, :]
        sc_frame2_atom = sc_trans[:, :, 2, :]
        sc_frame3_atom = sc_trans[:, :, 3, :]

        RBF_all = []
        RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame0_atom, E_idx)) #N-N
        RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame1_atom, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame2_atom, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(sc_frame0_atom, sc_frame3_atom, E_idx)) #Ca-C

        RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame0_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame1_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame2_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame1_atom, sc_frame3_atom, E_idx)) #C-C

        RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame0_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame1_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame2_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame2_atom, sc_frame3_atom, E_idx)) #C-C

        RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame0_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame1_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame2_atom, E_idx)) #C-C
        RBF_all.append(self._get_rbf(sc_frame3_atom, sc_frame3_atom, E_idx)) #C-C
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all
        

    def torsion_to_sidechain_frames(self, torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype):
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
        
        # atom14_atom_exists = restype_atom14_mask.to(device)[aatype]
        # import pdb; pdb.set_trace()
        # # for debug sidechain
        # aatype = aatype.reshape(B, L)
        # residx_atom37_to_atom14 = restype_atom37_to_atom14.to(device)[aatype]
        # residx_atom37_mask = restype_atom37_mask.to(device)[aatype] 
        # atom37 = atom14_to_atom37_batch(rec_atom14_tensor, residx_atom37_to_atom14, residx_atom37_mask)
        # debug_f = f'/raw22/superbrain/permanent/yfliu25/ligand_protdesign/protenc_ligenc_enc_dec_pifold_all_lig/experiments/debug/debug_f/batch_{0}_disc10_gt_torsion0.pdb'
        # f_ctx = write_coords(atom37[0], aatype[0], residx_atom37_mask[0], debug_f)

        return tensor_flat12, rec_atom14_tensor

        


class SideChainDecoder(nn.Module):
    def __init__(self, torsion_bins, h_VES_dim, n_embd, n_layer, n_head, dropout=0.0, torsion_noise_degree=5.) -> None:
        super().__init__()
        self.prediction_head_num = 4
        self.torsion_bins = torsion_bins
        
        self.activate_h_VES = nn.Linear(h_VES_dim, n_embd)

        self.sidechain_embedders = nn.ModuleList()
        for _ in range(self.prediction_head_num-1): # no last
            self.sidechain_embedders.append(
                nn.Embedding(self.torsion_bins, n_embd))

        noise_rad = np.deg2rad(torsion_noise_degree)
        self.vonmises_noise_dist = torch.distributions.VonMises(torch.tensor([0.]), torch.tensor([1.0/noise_rad]))
        
        depth_model_args = dict(
            vocab_size=1, # fake vocab
            n_positions = self.prediction_head_num,
            n_ctx=4, n_embd=n_embd,
            n_layer=n_layer, n_head=n_head, 
            rotary_dim=None, 
            gradient_checkpointing=True,
            embd_pdrop=dropout)
        depth_config = ProGenConfig(**depth_model_args)
        self.model = ProGenModel(depth_config)
        
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.prediction_head_num):
            self.prediction_heads.append(
                nn.Linear(self.model.embed_dim, torsion_bins) )


    def forward(self, h_VES, torsion_angle, torsion_angle_mask):
        act_h_VES = self.activate_h_VES(h_VES) # B, L, D

        # chi_noise = self.vonmises_noise_dist(torsion_angle.shape)[..., 0]
        # if self.training:
        #     torsion_angle = torsion_angle + chi_noise
        disc_torsion_angle = discrete_torsion(torsion_angle, (-np.pi, np.pi, self.torsion_bins))

        bsz, res_num = disc_torsion_angle.shape[:2] # B, L, 4
        inputs_embeds = [act_h_VES[:, :, None]] # B, L, 1, D

        for sc_idx, sc_embedder in enumerate(self.sidechain_embedders): 
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
        for pred_idx, pred_head in enumerate(self.prediction_heads): 
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
                cur_sc_embeds = self.sidechain_embedders[sc_idx-1](last_sc_disc_torsion_angle) * torsion_angle_mask[..., sc_idx-1][..., None] # B, L, D
                raw_inputs_embeds.append(cur_sc_embeds[:, :, None])

            inputs_embeds = torch.cat(raw_inputs_embeds, 2) # B, L, X, D
            inputs_embeds = inputs_embeds.reshape(bsz*res_num, sc_idx+1, -1) # BxL, X, D

            if self.training:
                use_gradient_checkpoint = True
            else:
                use_gradient_checkpoint = False

            dec_output = self.model(inputs_embeds = inputs_embeds,  input_bias = 0.0, input_bias_layer = -1, use_gradient_checkpoint = use_gradient_checkpoint)
            dec_output = dec_output[0][:, -1].reshape(bsz*res_num, -1) # BxL, D

            cur_pred_torsion_angle_logits = self.prediction_heads[sc_idx](dec_output) # BxL, D
            logits = cur_pred_torsion_angle_logits / temperature
            probs = F.softmax(logits, dim=-1)  # B, L, D
            last_sc_disc_torsion_angle = torch.multinomial(probs, num_samples=1).reshape(bsz, res_num) # B, L
            stacked_sc_disc_torsion_angle.append(last_sc_disc_torsion_angle[..., None])

            raw_cur_nll = torch.log(torch.gather(probs.reshape(bsz, res_num, -1), -1, last_sc_disc_torsion_angle[..., None])[..., 0])
            stacked_sc_disc_torsion_angle_prob.append(raw_cur_nll)

        return torch.cat(stacked_sc_disc_torsion_angle, -1), torch.stack(stacked_sc_disc_torsion_angle_prob, -1)


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16, embed_prot_lig_edge=False):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
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
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
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
        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        edge_type = (1 - mask)[:,:,None] * (1 - mask)[:,None, :] * 3
        if (lig_mask is not None):
            edge_type = (lig_mask[:,:,None] + lig_mask[:,None,:]) + edge_type
            # edge_type = (lig_mask[:,:,None] + lig_mask[:,None,:]) + (1 - mask)[:,:,None] * (1 - mask)[:,None,:] * 3
            # E_edge_type = gather_edges(edge_type[:,:,:,None], E_idx)[:,:,:,0]
            # E_type_embed = self.edge_type_embedding(E_edge_type.long()) # 0: protprot; 1: prot-lig; 2: lig-lig; 3: mask
            # E = E + E_type_embed
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
        ligmpnn_init=False, esm_embedder=False, max_iter_num=4, embed_unimol_reprs=False, 
        torsion_angle_gap=10, tosion_decoder_layer=3, tosion_decoder_head=4, 
        torsion_loss_weight=0.6, avail_torsion_loss_weight=0.15, encode_mpnn=False):
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

        if pre_prot_embed:
            if encode_mpnn:
                self.prot_encoder_mpnn = ProteinMPNNEncoder(
                    node_features, edge_features, hidden_dim, k_neighbors=48, augment_eps=0.2)
            self.prot_encoder_pifold = PiFoldEncoder()

        if esm_embedder:
            self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet(ESM_PRETRAIN)
            self.esm_model_embedder = nn.Sequential(
                nn.Linear(1280, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim))
            # self.last_iter_W_s = nn.Embedding(vocab, hidden_dim)
            self.traj_loss_weight = 0.5

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
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, embed_prot_lig_edge=True)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_c = TorsionEmbedding(self.torsion_bins, hidden_dim, hidden_dim, embedding_scatoms=True)
        # self.activate_W_sc_node = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.activate_W_sc_node = nn.Sequential(
        #     nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
        #     nn.LeakyReLU(),
        #     BatchNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim, bias=True),
        #     nn.LeakyReLU(),
        #     BatchNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim, bias=True)
        # )
        self.activate_W_sc_edge = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        # self.activate_W_sc_merge_edge = nn.Sequential(
        #     nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=True)
        # )
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

        if esm_embedder:
            self._freeze_esm_parameter()


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


    def _freeze_esm_parameter(self, ):
        for name, param in self.esm_model.named_parameters():
            param.requires_grad = False
        logger.info(f'freeze parameters of esm model')


    def forward(
        self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, 
        lig_node_attr, lig_edge_attr, lig_edge_index, 
        use_input_decoding_order=False, decoding_order=None,lig_mask=None, seq_mask=None, 
        esm_embedding=None, last_iter_S=None, unimol_reprs=None, 
        torsion_angles=None, alt_chi_angles=None, torsion_angle_mask=None, 
        backbone_affine_tensor=None, backbone_angles_sin_cos=None):
        """ Graph-conditioned sequence model """

        device=X.device
        B, L = X.shape[:2]
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        chain_M = chain_M*mask #update chain_M to include missing regions
        torsion_angle_mask = torsion_angle_mask * prot_mask[..., None]
        # import pdb; pdb.set_trace()

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

        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)
        
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        if self.esm_embedder:
            if (esm_embedding is not None):
                pre_iter_h_V = self.esm_model_embedder(esm_embedding) * prot_mask[..., None].to(E.device)
            else:
                fake_pre_h_V = torch.zeros((E.shape[0], E.shape[1], 1280), device=E.device)
                pre_iter_h_V = self.esm_model_embedder(fake_pre_h_V) * prot_mask[..., None].to(E.device)
            h_V = h_V + pre_iter_h_V

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

            if self.lig_neighbor_seq_mask:
                avail_unmask = seq_mask.float() * prot_mask
            else:
                avail_unmask = prot_mask
            # DEBUG: prev_tokens = (S * (1 - avail_unmask)).long()
            prev_tokens = inject_noise(S, avail_unmask.bool(), noise=self.noise)

            sc_mask = (torch.all(torch.stack([(prev_tokens >= 4), prev_tokens!=24], -1), -1)).float()
            prev_torsion = sc_mask[..., None] * torsion_angles
            prev_alt_torsion = sc_mask[..., None] * torsion_angles
            prev_torsion_mask = sc_mask[..., None] * torsion_angle_mask

            if self.esm_embedder:
                prev_tokens = (1 - prot_mask + lig_mask * 23).long()

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(prev_tokens)
        if self.esm_embedder:
            if (last_iter_S is not None):
                h_S = h_S + self.W_s(last_iter_S)

        sc_mask_attend = gather_nodes(sc_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        h_C, h_EC_ = self.W_c(prev_torsion, prev_torsion_mask, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype=prev_tokens, mask_attend=sc_mask_attend)
        h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1))
        h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES
        # h_ESC = activate_W_sc_merge_edge(h_ESC)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx)
            h_ESV = mask_bw * (h_ESCV) + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        
        ## teacher forcing training p(sc^i | aa^i, aa^j, sc^j) * p(aa^i | aa^j, sc^j) = p(aa^i, sc^i | aa^j, sc^j)
        all_h_S = self.W_s(S)
        sc_res_ctx = torch.cat([all_h_S, h_V], -1)
        sc_logits = self.sc_decoder(sc_res_ctx, torsion_angles, torsion_angle_mask)
        log_sc_probs = F.log_softmax(sc_logits, dim=-1)

        return log_probs, prev_tokens, log_sc_probs, prev_torsion_mask


    def nar_sample(self, X, S, mask, chain_M, residue_idx, chain_encoding_all,
        lig_node_attr, lig_edge_attr, lig_edge_index, 
        lig_mask, iter_num=5, temperature=1.0, seq_mask_pocket=None, 
        backbone_affine_tensor=None, backbone_angles_sin_cos=None,
        esm_embedding=None, unimol_reprs=None, initial_S=None,
        esm_batcher=None, esm_model=None, esm_alphabet=None, mask_mode='sidechain_nll'):
        
        device=X.device
        B, L = X.shape[:2]
        esm_embedding = None
        res_num = X.shape[-1]
        bsz = X.shape[0]
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        chain_M = chain_M*mask #update chain_M to include missing regions

        initial_S_ = S.clone().float()   
        initial_S_pocket = initial_S_ * (1 - seq_mask_pocket.float()) + lig_mask * 24 + (1 - mask) 
        
        if initial_S is None:
            initial_S = initial_S_ * (1 - prot_mask) 
            given_initial_S = False
        else:
            given_initial_S = True
            

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

        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)
        
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        if self.esm_embedder:
            if (esm_embedding is not None):
                pre_iter_h_V = self.esm_model_embedder(esm_embedding) * prot_mask[..., None].to(E.device)
            else:
                fake_pre_h_V = torch.zeros((E.shape[0], E.shape[1], 1280), device=E.device)
                pre_iter_h_V = self.esm_model_embedder(fake_pre_h_V) * prot_mask[..., None].to(E.device)
            h_V = h_V + pre_iter_h_V

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
        if not given_initial_S:
            init_iter_mask = cur_iter_masked_S == 0
        else:
            init_iter_mask = prot_mask == 1
        init_iter_mask_pocket = initial_S_pocket == 0

        cur_iter_S = None
        iter_seq_list = []
        iter_identity_list = []
        iter_identity_list_pocket = []
        # given pre-designed torsion information in cur_iter_torsion
        cur_iter_torsion = torch.zeros((B, L, 4)).float().to(device)
        cur_iter_torsion_mask = torch.zeros((B, L, 4)).float().to(device)
        # if side-chain packing only, then cur_iter_torsion is None
        for cur_iter_num in range(iter_num):
            cur_iter_mask = cur_iter_masked_S == 0
            # Concatenate sequence embeddings for autoregressive decoder
            h_S = self.W_s(cur_iter_masked_S.long())

            sc_mask = (torch.all(torch.stack([(cur_iter_masked_S >= 4), cur_iter_masked_S!=24], -1), -1)).float()
            cur_iter_torsion = sc_mask[..., None] * cur_iter_torsion
            cur_iter_torsion_mask = sc_mask[..., None] * cur_iter_torsion_mask
            sc_mask_attend = gather_nodes(sc_mask.unsqueeze(-1),  E_idx).squeeze(-1)

            h_C, h_EC_ = self.W_c(cur_iter_torsion, cur_iter_torsion_mask, E_idx, backbone_affine_tensor, backbone_angles_sin_cos, aatype=cur_iter_masked_S.long(), mask_attend=sc_mask_attend)
            h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1))
            h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
            # mask_bw = mask_1D * mask_attend
            mask_bw = mask_1D * (1. - mask_attend)
            mask_fw = mask_1D * (1. - mask_attend)

            # Build encoder embeddings
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
            raw_S_iter, raw_S_iter_score = sample_from_categorical(logits, temperature)

            if torch.any(cur_iter_mask):
                cur_iter_S = (cur_iter_mask.float() * raw_S_iter + cur_iter_masked_S).long()
            else:
                cur_iter_S = raw_S_iter.long()

            if (esm_batcher is not None):
                seq_list = []
                for b_aatype in (cur_iter_S * prot_mask).cpu().numpy():
                    sequence = ''.join([res_id_to_aatype[res_id] for res_id in b_aatype if (res_id not in [0, 1, 2, 3])])
                    seq_list.append(sequence)

                prot_aa_num = len(seq_list[0])
                refined_seqs, refined_tokens, refined_raw_cur_nll = esm_refine(seq_list, esm_batcher, esm_model, esm_alphabet, device)
                cur_iter_S[:, :prot_aa_num] = refined_tokens

                output_scores = (1 - init_iter_mask.float()) * 2# + refined_raw_cur_nll * init_iter_mask.float()
                output_scores[:, :prot_aa_num] = refined_raw_cur_nll
            else:
                raw_cur_nll = torch.log(torch.gather(probs, -1, raw_S_iter[..., None])[..., 0])
                output_scores = (1 - init_iter_mask.float()) * 2 + raw_cur_nll * init_iter_mask.float()

            ## teacher forcing training p(sc^i | aa^i, aa^j, sc^j) * p(aa^i | aa^j, sc^j) = p(aa^i, sc^i | aa^j, sc^j)
            all_h_S = self.W_s(cur_iter_S)
            sc_res_ctx = torch.cat([all_h_S, h_V], -1)

            cur_iter_torsion_mask = make_torsion_mask_from_aatype(cur_iter_S) # cur_iter_S
            cur_iter_disc_torsion, cur_iter_disc_torsion_nll = self.sc_decoder.sample(sc_res_ctx, cur_iter_torsion_mask, temperature)
            cur_iter_torsion = self.W_c.disc_to_conti.to(device)[cur_iter_disc_torsion]

            iter_seq_list.append(cur_iter_S)

            cur_batch_iter_ident = ((cur_iter_S == S).float() * init_iter_mask.float()).sum(-1)/(init_iter_mask.float() + 1e-10).sum(-1)
            iter_identity_list.append(cur_batch_iter_ident) # all identity
            cur_batch_iter_ident_pocket = ((cur_iter_S == S).float() * init_iter_mask_pocket.float()).sum(-1)/(init_iter_mask_pocket.float() + 1e-10).sum(-1)
            iter_identity_list_pocket.append(cur_batch_iter_ident_pocket) # pocket identity

            output_scores_mask = init_iter_mask
            # cur_iter_p = 0.1
            cur_iter_p = 1 - (cur_iter_num + 1) / iter_num
            if (mask_mode == 'aatype_nll'):
                iter_output_scores = output_scores
            elif (mask_mode == 'sidechain_nll'):
                iter_output_scores = cur_iter_disc_torsion_nll.mean(-1)
            else:
                raise ValueError
            skeptical_mask = _skeptical_unmasking(iter_output_scores, output_scores_mask, cur_iter_p)
            cur_iter_masked_S = cur_iter_S.masked_fill(skeptical_mask, 0.0)

            h_V = h_V_init
            h_E = h_E_init

        iter_seq_list = torch.stack(iter_seq_list)
        iter_identity_list = torch.stack(iter_identity_list)
        iter_identity_list_pocket = torch.stack(iter_identity_list_pocket)
        pdb_ctx = make_pdb_ctx_from_design(iter_seq_list[-1], cur_iter_torsion, backbone_angles_sin_cos, backbone_affine_tensor)

        design_ctx = {
            'aatype': iter_seq_list.cpu(),
            'all_identity': iter_identity_list.cpu(),
            'pocket_identity': iter_identity_list_pocket.cpu(),
            'protein_mask': prot_mask,
            'pdb_ctx': pdb_ctx
            # 'all_nll': iter_nll_list.cpu()
        }

        return design_ctx


    def ar_sample(self,  X, S, mask, chain_M, residue_idx, chain_encoding_all,randn,
        lig_X, lig_node_attr, lig_edge_attr, lig_edge_index, 
        lig_mask, seq_mask, temperature=1.0, seq_mask_poeckt=None, unimol_reprs=None):
        
        device=X.device
        N_batch, N_nodes = X.size(0), X.size(1)
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
        chain_M = chain_M*mask #update chain_M to include missing regions

        S_true = S
        initial_S_ = S.clone().float()
        if self.lig_neighbor_seq_mask:
            initial_S = initial_S_ * (1 - seq_mask.float()) + lig_mask * 24 + (1 - mask)  
        else:
            initial_S = initial_S_ * (1 - prot_mask) 
        initial_S_pocket = initial_S_ * (1 - seq_mask_poeckt.float()) + lig_mask * 24 + (1 - mask)
        init_iter_mask = initial_S == 0
        init_iter_mask_pocket = initial_S_pocket == 0
        init_iter_mask_non_pocket = (init_iter_mask.float() - init_iter_mask_pocket.float()).bool()

        if self.pre_prot_embed:
            # Prepare node and edge embeddings
            prot_X = X * prot_mask[..., None, None]
            pre_h_V_encoder = self.prot_encoder(prot_X, initial_S, prot_mask, residue_idx, chain_encoding_all)
        else:
            pre_h_V_encoder = 0

        lig_X = X[:, :, 1] * lig_mask[..., None]
        lig_node = self.lig_encoder(lig_node_attr, lig_X, lig_edge_index, lig_edge_attr)

        if self.embed_unimol_reprs:
            unimol_reprs = self.unimol_repr_activate(unimol_reprs) * lig_mask[..., None]
            lig_node = lig_node + unimol_reprs

        # # for validation mask all lig inf
        # lig_mask = torch.zeros_like(lig_mask)
        # mask = prot_mask

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

        # Decoder uses masked self-attention
        chain_mask = chain_M #update chain_M to include missing regions

        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

        if self.lig_neighbor_seq_mask:
            unmasked_seq = 1 - seq_mask.float()
            order_mask_backward = order_mask_backward * unmasked_seq[:, :, None]
        
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        all_probs = torch.zeros((N_batch, N_nodes, 35), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)

        h_S = self.W_s(initial_S.long()) * (initial_S != 0).float()[..., None]
        S = torch.zeros((N_batch, N_nodes), device=device).float() + initial_S
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            
            if (initial_S[0][t][0] != 0):
                continue
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits, dim=-1)
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,35), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
                S_t = (S_t*chain_mask_gathered).long()
                h_S_t = self.W_s(S_t)
                h_S.scatter_(1, t[:,None,None].repeat(1,1,h_S_t.shape[-1]), h_S_t)
                S.scatter_(1, t[:,None], S_t.float())
        
        # output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        iter_seq_list = S
        cur_batch_iter_ident = ((S == S_true).float() * init_iter_mask.float()).sum(-1)/(init_iter_mask.float() + 1e-10).sum(-1)
        iter_identity_list = cur_batch_iter_ident # all identity
        cur_batch_iter_ident_pocket = ((S == S_true).float() * init_iter_mask_pocket.float()).sum(-1)/(init_iter_mask_pocket.float() + 1e-10).sum(-1)
        iter_identity_list_pocket = cur_batch_iter_ident_pocket # pocket identity
        cur_batch_iter_ident_non_pocket = ((S == S_true).float() * init_iter_mask_non_pocket.float()).sum(-1)/(init_iter_mask_non_pocket.float() + 1e-10).sum(-1)
        iter_identity_list_non_pocket = cur_batch_iter_ident_non_pocket # non-pocket identity
        # iter_nll_list = -torch.log(all_probs)

        design_ctx = {
            'aatype': iter_seq_list[None].cpu(),
            'all_identity': iter_identity_list[None].cpu(),
            'pocket_identity': iter_identity_list_pocket[None].cpu(),
            'non_pocket_identity': iter_identity_list_non_pocket[None].cpu(),
            # 'all_nll': iter_nll_list.cpu()
        }

        return design_ctx


    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None,\
         chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions

        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]
            if (chain_mask_gathered==0).all():
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def tied_sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, \
        chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None,\
             tied_pos=None, tied_beta=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(X.shape[0],1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (chain_mask[:,t]==0).all():
                    S_t = S_true[:,t]
                    for t in t_list:
                        h_S[:,t,:] = self.W_s(S_t)
                        S[:,t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:,t:t+1,:]
                    h_E_t = h_E[:,t:t+1,:,:]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
                    mask_t = mask[:,t:t+1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:,t:t+1,:]
                        h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:,t,:]
                    logit_list.append((self.W_out(h_V_t) / temperature)/len(t_list))
                    logits += tied_beta[t]*(self.W_out(h_V_t) / temperature)/len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:,t,:] #[B, 21]
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:,t]
                    pssm_bias_gathered = pssm_bias[:,t]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:,t]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                for t in t_list:
                    h_S[:,t,:] = self.W_s(S_t_repeat)
                    S[:,t] = S_t_repeat
                    all_probs[:,t,:] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
  
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0,:]==1)[:,0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:,idx,:] = log_probs[:,idx,:]
        return log_conditional_probs


    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


