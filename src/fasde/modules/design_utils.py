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
from .pifold_encoder import PiFoldEncoder
from .pifold.batch_module import StructureEncoder
from .cmlm_mask import inject_noise
from .progen.modeling_progen import ProGenModel
from .progen.configuration_progen import ProGenConfig
from .save_all_atoms import write_coords
from fasde.utils.relax.assess_violation import get_violation_metrics
# from assess_violation import get_violation_metrics

import sys
from .alphafold import all_atom, r3, quat_affine
from .alphafold.model2.folding import atom14_to_atom37_batch
from .adaln import AdaLNDecLayer, CrossAttnDecLayer
from .normal_encoder_single_rbf import NormalEncoder as AdaLNNormalEncoder
from .alphafold.data.utils.data_transforms import restype_atom37_mask, restype_atom37_to_atom14, restype_atom14_mask
from .alphafold.common import residue_constants
from .alphafold.common.protein import from_pdb_string

from esm import pretrained

from .normal_encoder_single_rbf import NormalEncoder  #对膜的位置进行编码，之后直接嵌入节点中
from .normal_encoder_single_rbf import virtual_cb_from_backbone, interface_offset_rbf_encoding
import traceback
import logging
logger = logging.getLogger(__file__)
logger.name = "design_utils"

DEBUG = False
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

torch.set_printoptions(threshold=float('inf'))


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




#     return esm_toks.long()
def converter_from_af2_to_esm(pred_merged_aatype, prot_mask):
    device = pred_merged_aatype.device
    af2_to_esm_indices_on_device = af2_to_esm_convert_indices.to(device)

    esm_no_bos_toks = []
    for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
        b_esm_tokens = af2_to_esm_indices_on_device[b_af2_tokens]
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
    """给定未归一化的 logit 向量（任意形状），按温度采样或贪婪解码得到离散 token 及其对应 log-probability。

    Args:
        logits (_type_, optional): _description_. Defaults to None.
        temperature (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    #如果给了温度，则按照相应的温度进行采样
    if temperature:

        #统一除以T，t越大，softmax之后采样越均匀
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        #如果没有给温度，则直接贪婪采样
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    """返回在neighbor——idx上的nodes节点特征

    Args:
        nodes (_type_): 一个张量[B,N,C]
        neighbor_idx (_type_): 另一个张量[B,N,K]

    Returns:
        _type_: 一个新的张量[B,N,K,C],意义是在每个节点上的所有邻居的边的特征
    """
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))#展开成为二维张量变为[B,N*K]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))#[B,N*K,C]
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)#[B,N*K,C]
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])#[B,N,K,C]
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)#[B,N,K,C]+[B,N,K,C']
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
    """
    它根据氨基酸类型（aatype，整数编码）返回一个 “每个残基每个可旋转二面角是否存在” 的布尔/浮点掩码张量，形状 [B, N, 4]，
    用于后续构象采样或损失计算。
    """
    aatype = aatype - 4
    # aatype.shape [B, N]
    B, N = aatype.shape
    device = aatype.device
    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    # 这是一个固定表：[restype, 4]，表示“该氨基酸的第 1–4 个 χ 角是否存在”。

    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.FloatTensor(chi_angles_mask).to(aatype.device)
    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    indices = aatype.unsqueeze(-1).repeat(1, 1, 4).view(-1, 4)

    if DEBUG:
        import pdb;pdb.set_trace()
    try:
        chis_mask = torch.gather(chi_angles_mask, 0, indices).view(B, N, 4)
    except Exception as e:
        traceback.print_exc()
        import pdb;pdb.set_trace()

    # bb_torsion_mask = torch.ones((B, N, 3))
    # all_torsion_mask = torch.cat([bb_torsion_mask, chis_mask], -1)

    return chis_mask


def get_violation_mask_from_design(aatype, torsion_angle, backbone_angles_sin_cos, backbone_affine_tensor):
    """把“设计出的序列 + 主链/侧链扭转角 + 刚性体”还原成全原子坐标，再用 Rosetta-style 的立体化学检查 找出 几何冲突（clash、键长键角、二面角异常等），返回一个 布尔掩码，标记哪些残基存在 结构性违规（violation）。

    Args:
        aatype (_type_): _description_
        torsion_angle (_type_): _description_
        backbone_angles_sin_cos (_type_): _description_
        backbone_affine_tensor (_type_): _description_

    Returns:
        _type_: _description_
    """
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
        """_summary_

        Args:
            h_V (_type_): B,N,C
            h_E (_type_): B,N,K,C
            E_idx (_type_): B,N,K
            mask_V (_type_, optional): _description_. Defaults to None.
            mask_attend (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)#将hv进行图传播，并且加上相应的h_e，返回的h_EV的维度是B,N,K,C'
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)#扩展HV，使其的维度变成B,N,K,C
        h_EV = torch.cat([h_V_expand, h_EV], -1)#在C维度上补充张量，维度变为[B,N,K,C+C']，相当于是做了一个残差
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))#进行MLP
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message#掩码
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
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))#重复了一遍信息传播
        h_E = self.norm3(h_E + self.dropout3(h_message))#将hmessage输入到新的E上并进行归一化
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


class MembraneNet(nn.Module):
    def __init__(self, hidden_dim, num_centers=20, tm_raw='raw', mem_between_mode='none', dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_centers = num_centers
        self.tm_raw = tm_raw
        self.mode = mem_between_mode
        self.use_tm_depth = 'depth' in tm_raw
        self.use_tm_theta = 'theta' in tm_raw
        self.use_membrane = self.use_tm_depth or self.use_tm_theta
        self.mem_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        valid_modes = {'none', 'add', 'concat'}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported mem_between_mode: {self.mode}")

        self.normal_encoder = None
        if self.use_membrane:
            self.normal_encoder = NormalEncoder(
                y_hidden_dim=128,
                y_output_dim=hidden_dim,
                theta_hidden_dim=128,
                theta_output_dim=hidden_dim,
                num_centers=self.num_centers,
                mem_between_mode=self.mode,
            )

        if self.mode == 'concat':
            self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
            nn.init.zeros_(self.concat_proj.weight)
            nn.init.zeros_(self.concat_proj.bias)

    def _build_membrane_embedding(self, X, G, N, prot_mask):
        result = self.normal_encoder(X, G, N)
        y_emb = result["y_emb"]
        theta = result["abs_depth"]

        if not torch.is_tensor(y_emb) or y_emb.dim() != 3:
            raise RuntimeError(f"NormalEncoder returned invalid y_emb: {type(y_emb)}, shape={getattr(y_emb, 'shape', None)}")
        if y_emb.shape[:2] != prot_mask.shape or y_emb.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"y_emb shape mismatch, expected [B, L, {self.hidden_dim}] with [B, L]={tuple(prot_mask.shape)}, got {tuple(y_emb.shape)}"
            )

        mem_V = torch.zeros(
            prot_mask.shape[0],
            prot_mask.shape[1],
            self.hidden_dim,
            device=prot_mask.device,
            dtype=self.mem_norm.weight.dtype,
        )
        if self.use_tm_depth:
            mem_V = mem_V + y_emb.to(dtype=mem_V.dtype)
        if self.use_tm_theta:
            if not torch.is_tensor(theta) or theta.dim() != mem_V.dim():
                raise RuntimeError(
                    f"tm_raw requests theta but NormalEncoder returned invalid theta: type={type(theta)}, shape={getattr(theta, 'shape', None)}"
                )
            mem_V = mem_V + theta.to(dtype=mem_V.dtype)
        return mem_V * prot_mask[..., None]

    def forward(self, h_V, X, G, N, prot_mask):
        if not self.use_membrane or self.mode == 'none':
            return h_V

        mem_V = self._build_membrane_embedding(X, G, N, prot_mask)

        mask = prot_mask[..., None]
        mem_V = self.mem_norm(mem_V).to(dtype=h_V.dtype) * mask
        if not torch.isfinite(mem_V).all():
            raise RuntimeError("Membrane embedding after LayerNorm contains NaN or Inf")

        if self.mode == 'add':
            return h_V + self.dropout(mem_V)
        if self.mode == 'concat':
            fused = torch.cat([h_V, self.dropout(mem_V)], dim=-1)
            return h_V + self.dropout(self.concat_proj(fused)) * mask

        return h_V


class MemEdgeEncoder(nn.Module):
    """把膜的深度(depth)+角度(theta)信息过一个基础 MLP 映射到 out_dim 维的逐残基嵌入,
    之后广播到 decoder 的边特征 h_ESV 后面一起随 decoder 传输。"""
    def __init__(self, out_dim=16, hidden_dim=32, num_centers=20,
                 abs_depth_noise_std=0.5, theta_noise_std_degrees=10.0):
        super().__init__()
        self.eps = 1e-6
        self.out_dim = int(out_dim)
        self.num_centers = int(num_centers)
        self.abs_depth_noise_std = float(abs_depth_noise_std)
        self.theta_noise_std_degrees = float(theta_noise_std_degrees)
        # 输入 = [depth RBF(num_centers) | sin(theta), cos(theta)]
        self.mlp = nn.Sequential(
            nn.Linear(self.num_centers + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, X, G, N, prot_mask):
        if G is None or N is None:
            return torch.zeros(X.shape[0], X.shape[1], self.out_dim, device=X.device, dtype=X.dtype)
        G = G.to(device=X.device, dtype=X.dtype)
        N = N.to(device=X.device, dtype=X.dtype)
        gr = G[:, :, :3]
        gt = G[:, :, 3:]
        ca = X[:, :, 1, :]
        r_mem = torch.bmm(gr, (ca + gt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
        half_thickness = N.norm(dim=-1).clamp(min=self.eps)
        unit_normal = N / half_thickness.unsqueeze(-1)
        signed_depth = (r_mem * unit_normal.unsqueeze(1)).sum(dim=-1)
        abs_depth = signed_depth.abs()
        if self.training and self.abs_depth_noise_std > 0:
            abs_depth = (abs_depth + torch.randn_like(abs_depth) * self.abs_depth_noise_std).clamp_min(0.0)
        interface_offset = abs_depth - half_thickness.unsqueeze(1)
        n_atom = X[:, :, 0, :]
        c_atom = X[:, :, 2, :]
        virtual_cb = virtual_cb_from_backbone(n_atom, ca, c_atom)
        ca_to_cb = virtual_cb - ca
        ca_to_cb = ca_to_cb / ca_to_cb.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        outward_sign = torch.where(signed_depth >= 0, torch.ones_like(signed_depth), -torch.ones_like(signed_depth))
        outward_normal = unit_normal.unsqueeze(1) * outward_sign.unsqueeze(-1)
        cos_theta = (ca_to_cb * outward_normal).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        if self.training and self.theta_noise_std_degrees > 0:
            theta_noise_std = theta.new_tensor(self.theta_noise_std_degrees * 3.141592653589793 / 180.0)
            theta = (theta + torch.randn_like(theta) * theta_noise_std).clamp(min=0.0, max=3.141592653589793)
        depth_rbf = interface_offset_rbf_encoding(interface_offset, num_centers=self.num_centers)
        theta_feats = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        feats = torch.cat([depth_rbf, theta_feats], dim=-1)
        emb = self.mlp(feats)
        is_null = half_thickness < 1e-3
        if is_null.any():
            emb = emb.clone()
            emb[is_null] = 0.0
        return emb * prot_mask[..., None]


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        """首先根据class中定义的maxrelativefeature = 32来将同一条链之间的氨基酸放缩成0~2*max..之间的索引，之后加上chain的掩码，如果不一样，直接加上2倍的max
        这里看看能不能做一个调整，如果不是在一条链的话直接设置成一个更大的数字

        Args:
            offset (_type_): 氨基酸之间的差值矩阵
            mask (_type_): 一个chain的掩码（布尔值？）标记两个氨基酸是不是在同一条链内


        Returns:
            _type_: _description_
        """
        # d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        dtype = next(self.parameters()).dtype   # 取当前模块权重的 dtype
        E = self.linear(d_onehot.to(dtype))
        return E


class TorsionEmbedding(nn.Module):
    def __init__(self, torsion_bins, single_torsions_dim, out_dim, ops='cat',
                    torsion_noise_degree=5., embedding_scatoms=False, num_rbf=16):
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
        # 对扭转角进行离散化操作 B, L, 4
        assert (len(disc_torsion_angle.shape) == 3) # B, L, 4

        # chi1_embedding = self.chi1_embedder(disc_torsion_angle[..., 0]) * torsion_angle_mask[..., 0][..., None]
        # chi2_embedding = self.chi2_embedder(disc_torsion_angle[..., 1]) * torsion_angle_mask[..., 1][..., None]
        # chi3_embedding = self.chi3_embedder(disc_torsion_angle[..., 2]) * torsion_angle_mask[..., 2][..., None]
        # chi4_embedding = self.chi4_embedder(disc_torsion_angle[..., 3]) * torsion_angle_mask[..., 3][..., None]

        #全都是embedding
        chi1_embedding = self.chi1_embedder_(disc_torsion_angle[..., 0]) * torsion_angle_mask[..., 0][..., None]
        chi2_embedding = self.chi2_embedder_(disc_torsion_angle[..., 1]) * torsion_angle_mask[..., 1][..., None]
        chi3_embedding = self.chi3_embedder_(disc_torsion_angle[..., 2]) * torsion_angle_mask[..., 2][..., None]
        chi4_embedding = self.chi4_embedder_(disc_torsion_angle[..., 3]) * torsion_angle_mask[..., 3][..., None]
        if (self.ops == 'cat'):
            reduced_chi_embedding = torch.cat([chi1_embedding, chi2_embedding, chi3_embedding, chi4_embedding], -1)
        elif (self.ops == 'add'):
            reduced_chi_embedding = chi1_embedding + chi2_embedding + chi3_embedding + chi4_embedding
        #分别对四个扭转角进行embedding之后，将他们加起来或者进行拼接
        act_chi_embedding = self.activate_torsion(reduced_chi_embedding)
        #进行了一个MLP

        if self.embedding_scatoms:
            disc_torsion_angle_conti = self.disc_to_conti.to(device)[disc_torsion_angle]

            #这里生成的是两个氨基酸之间的每个侧链主要原子之间的距离编码，实际上已经包含了相互作用信息，如果在生成氨基酸或者扭转角的时候使用就是泄露
            sc_rbf_edge = self.sidechain_dist_embed(disc_torsion_angle_conti, E_idx, aatype, backbone_coords, mask, lig_mask, mask_scatom) * mask_attend[..., None]

            #mlp激活
            # sc_rbf_edge_act = self.sc_rbf_edge_act(sc_rbf_edge)
            sc_rbf_edge_act = self.sc_rbf_edge_act_(sc_rbf_edge)

            #最后将每个节点的嵌入和之间的扭转角关系进行消息传递
            h_EC = cat_neighbors_nodes(act_chi_embedding, sc_rbf_edge_act, E_idx) * mask_attend[..., None]

            return act_chi_embedding, h_EC #这里还没有加上掩码

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
        """
        计算A与B之间的RBF特征，可选择性地应用掩码。

        参数:
            A (Tensor): 形状 [B, L, C_A] —— 批次大小 B，序列长度 L，通道数 C_A
            B (Tensor): 形状 [B, L, C_B] —— 批次大小 B，序列长度 L，通道数 C_B
            E_idx (Tensor): 形状 [B, L, K] —— 每个位置 i 的 K 个邻居的索引
            A_mask (Tensor, optional): 形状 [B, L] —— A 的掩码，1 表示有效位置
            B_mask (Tensor, optional): 形状 [B, L] —— B 的掩码，1 表示有效位置

        返回:
            RBF_A_B (Tensor): 形状 [B, L, K, N_rbf] —— 应用掩码后的RBF特征，N_rbf 为 RBF 展开维度
        """
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
        # [B, L, K, N_rbf]


    def sidechain_dist_embed(self, torsion_angle, E_idx, aatype, backbone_coords, mask, lig_mask, mask_scatom):
        B, N = aatype.shape
        device = aatype.device
        raw_aatype = aatype
        aatype = torch.where(aatype-4 < 0, 20, aatype-4)

        backbone_affine_tensor, backbone_angles_sin_cos = get_rigid_groups(backbone_coords)

        tensor_flat12, rec_atom14_tensor = self.torsion_to_sidechain_frames(
                                                                        torsion_angle,
                                                                        backbone_affine_tensor,
                                                                        backbone_angles_sin_cos,
                                                                        aatype )
        sc_atom14_mask = restype_atom14_mask.to(device)[aatype]
        sc_atom14_mask = torch.where(raw_aatype[:, :, None] == 24, ligatom_atom14_mask[None, None].repeat(B, N, 1).to(device), sc_atom14_mask)
        sc_atom14_atomtype = residue_atomtype.to(device)[aatype]
        N_atom  = backbone_coords[:, :, 0, :]
        Ca_atom = backbone_coords[:, :, 1, :]
        C_atom  = backbone_coords[:, :, 2, :]
        O_atom  = backbone_coords[:, :, 3, :]



        RBF_all = []
        for bb_atom_idx, backbone_atom in enumerate([N_atom, Ca_atom, C_atom, O_atom]):
            if bb_atom_idx == 0:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 2
                bb_atom_mask = mask * (1 - lig_mask)
            elif bb_atom_idx == 1:
                bb_atomtype = torch.ones(N_atom.shape[0], N_atom.shape[1]).to(N_atom.device) * 1 #如果是ca的话，不去掉配体
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
                neighbor_atom_mask = sc_atom14_mask[:, :, neighbor_atom_id]# 获取侧链原子的坐标和掩码

                # 主链原子与所有侧链原子之间的距离编码为高维RBF特征
                RBF_all.append(self._get_rbf(backbone_atom, neighbor_atom, E_idx, bb_atom_mask, neighbor_atom_mask, bb_atomtype, neighbor_atomtype)) #N-N

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all


    def torsion_to_sidechain_frames(self, torsion_angle, backbone_affine_tensor, backbone_angles_sin_cos, aatype):
        """根据主链（backbone）构象和预测的侧链扭转角（torsion angles），生成侧链的局部坐标系（frames）以及侧链原子在全局坐标下的 3D 坐标。


        Args:
            torsion_angle	[B, L, 4]	每个残基的 4 个侧链扭转角（χ1~χ4）
            backbone_affine_tensor	[B, L, 4, 12]	主链的 SE(3) 变换矩阵（4×12）
            backbone_angles_sin_cos	[B, L, 3, 2]	主链的 3 个扭转角（φ, ψ, ω）的 sin/cos 表示
            aatype	[B, L]	每个残基的氨基酸类型（整数编码）
        Returns:
            tensor_flat12	[B, L, 8, 12]	每个残基的所有 8 个局部坐标系（frames）的 SE(3) 变换矩阵（包括主链 + 侧链）
            rec_atom14_tensor	[B, L, 14, 3]	每个残基的 14 个标准原子在全局坐标下的 3D 坐标（来自 OpenFold 的 atom14 表示）
        """
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

        return tensor_flat12, rec_atom14_tensor#, atom14_atom_exists




class SideChainDecoder(nn.Module):
    def __init__(self, torsion_bins, h_VES_dim, n_embd, n_layer, n_head, dropout=0.0, torsion_noise_degree=5.):
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
        #######################################################
        # self.training = False
        #########################################################
        # print(f"training:{self.training}")
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
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)#生成distogram，里面的元素是两个X之间的距离
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max#如果对应的mask不是1，则将distogram中的距离调整为最大值
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        #返回值为（B,N,K）为每个节点最近邻的K个节点，里面装的是距离值
        # e_idx的形状也为B,N,K，是每个被选取的节点的索引
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
        """_summary_

        Args:
            A (_type_):原子坐标
            B (_type_): 原子坐标
            E_idx (_type_): 形状B,N,K

        Returns:
            _type_: 返回为B,N,K，里面为每个节点（氨基酸）中不同类别原子之间的距离
        """
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels, lig_mask=None):
        """对于输入的X B,N,M,3 生成氨基酸层面的图结构表征，B,N,K,C里面包含两个氨基酸之间的 每个原子的距离rbf编码，索引信息，链信息，蛋白配体信息，并且经过了embedding和层归一化

        Args:
            X (_type_): _description_
            mask (_type_): _description_
            residue_idx (_type_): _description_
            chain_labels (_type_): _description_
            lig_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
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
        E_positional = self.embeddings(offset.long(), E_chains)#现在将氨基酸序列上的差距和chain的掩码统一进行编码

        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        #得到的E的结构为E,N,K,C，c中包含了各种原子之间的距离信息，氨基酸idx信息，以及chain的信息

        edge_type = (1 - mask)[:,:,None] * (1 - mask)[:,None, :] * 3
        if (lig_mask is not None):
            edge_type = (lig_mask[:,:,None] + lig_mask[:,None,:]) + edge_type
        E_edge_type = gather_edges(edge_type[:,:,:,None], E_idx)[:,:,:,0]
        E_type_embed = self.edge_type_embedding(E_edge_type.long()) # 0: protprot; 1: prot-lig; 2: lig-lig; 3: mask
        #etypeembed只在区分这个究竟是氨基酸之间的互作还是配体之间的互作还是氨基酸和配体之间的互作
        E = E + E_type_embed
        E = self.norm_edges(E)
        #进行layernorm之后返回，形状分别为B,N,K,C 和 B,N,K
        return E, E_idx


class ABACUST(nn.Module):
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
        num_centers = 20,
        tm_raw = "raw",
        mem_between_mode='none',
        depth_inject_method='none',
        depth_cond_dim=128,
        depth_num_rbf_centers=20,
        cross_attn_num_heads=4,
        cfg_enabled=False,
        cfg_dropout_prob=0.15,
        freeze_modules=None,
        mem_logit_modulation=False,
        mem_logit_gate_modulation=False,
        mem_logit_gate_zero_encoder=False,
        mem_logit_normal_encoder_zero_region_embedding=False,
        mem_edge_dim=0,
        ):
        super(ABACUST, self).__init__()

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
        self.num_centers = num_centers
        self.tm_raw = tm_raw  #是否禁用pdbtm的嵌入标记
        self.mem_between_mode = mem_between_mode
        self.mem_edge_dim = int(mem_edge_dim)
        self.mem_edge_encoder = MemEdgeEncoder(out_dim=self.mem_edge_dim) if self.mem_edge_dim > 0 else None
        self.depth_inject_method = depth_inject_method
        self.depth_cond_dim = depth_cond_dim
        self.cfg_enabled = cfg_enabled
        self.cfg_dropout_prob = cfg_dropout_prob
        self.mem_logit_modulation = mem_logit_modulation
        self.mem_logit_gate_modulation = mem_logit_gate_modulation
        self.mem_logit_gate_zero_encoder = mem_logit_gate_zero_encoder
        self.mem_logit_normal_encoder_zero_region_embedding = mem_logit_normal_encoder_zero_region_embedding
        self._use_tm_depth = 'depth' in self.tm_raw
        self._use_tm_theta = 'theta' in self.tm_raw
        self._use_tm_any = self._use_tm_depth or self._use_tm_theta or mem_logit_modulation or mem_logit_gate_modulation
        logger.info(f"self.tm_raw:{tm_raw}" )
        logger.info(f"self.mem_between_mode:{mem_between_mode}")
        logger.info(f"tm flags depth/theta: {self._use_tm_depth}/{self._use_tm_theta}")
        logger.info(f"mem_logit_gate_zero_encoder: {self.mem_logit_gate_zero_encoder}")
        logger.info(f"mem_logit_normal_encoder_zero_region_embedding: {self.mem_logit_normal_encoder_zero_region_embedding}")

        if pre_prot_embed:
            if encode_mpnn:
                self.prot_encoder_mpnn = ProteinMPNNEncoder(
                    node_features, edge_features, hidden_dim, k_neighbors=48, augment_eps=0.2)
            self.prot_encoder_pifold = PiFoldEncoder()

        self.W_esm_LN = nn.LayerNorm(1280)
        self.W_esm_act = nn.Linear(1280, hidden_dim)


        if (lig_embed_mode == 'linear'):
            self.lig_encoder = LinearEncoder(lig_h_dim, lig_hidden)
        elif (lig_embed_mode == 'schnet'):#实际上用的是这个
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
        self.mem_logit_gate_encoder_layers = None
        if mem_logit_gate_modulation:
            self.mem_logit_gate_encoder_layers = copy.deepcopy(self.encoder_layers)

        # Decoder layers
        # decoder 边输入维度 = 3*hidden + mem_edge_dim (膜信息 concat 到 h_ESV 后面)
        dec_edge_in = hidden_dim*3 + self.mem_edge_dim
        if self.depth_inject_method == 'adaln':
            self.decoder_layers = nn.ModuleList([
                AdaLNDecLayer(hidden_dim, dec_edge_in, cond_dim=depth_cond_dim, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
        elif self.depth_inject_method == 'cross_attn':
            self.decoder_layers = nn.ModuleList([
                CrossAttnDecLayer(hidden_dim, dec_edge_in, cond_dim=depth_cond_dim,
                                  num_heads=cross_attn_num_heads, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                DecLayer(hidden_dim, dec_edge_in, dropout=dropout)
                for _ in range(num_decoder_layers)
            ])
        if self.depth_inject_method != 'none':
            self.depth_encoder = AdaLNNormalEncoder(
                y_output_dim=depth_cond_dim,
                num_centers=depth_num_rbf_centers,
            )
        # self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_letters, bias=True)
        )
        self.mem_net = None
        self.mem_logit_bias = None
        self.mem_logit_normal_encoder = None
        self.mem_logit_gate = None
        self.mem_logit_distribution_projection = None
        if self._use_tm_any:
            self.mem_net = MembraneNet(
                self.hidden_dim,
                num_centers=self.num_centers,
                tm_raw=self.tm_raw,
                mem_between_mode=self.mem_between_mode,
                dropout=dropout,
            )
        if mem_logit_modulation or mem_logit_gate_modulation:
            self.mem_logit_normal_encoder = NormalEncoder(
                y_hidden_dim=128,
                y_output_dim=hidden_dim,
                theta_hidden_dim=128,
                theta_output_dim=hidden_dim,
                num_centers=self.num_centers,
                mem_between_mode='none',
                zero_region_embedding=self.mem_logit_normal_encoder_zero_region_embedding,
            )

        if mem_logit_modulation:
            self.mem_logit_bias = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, num_letters, bias=True),
            )
            nn.init.zeros_(self.mem_logit_bias[-1].weight)
            nn.init.zeros_(self.mem_logit_bias[-1].bias)

        if mem_logit_gate_modulation:
            self.mem_logit_gate = nn.Linear(hidden_dim * 2, 1, bias=True)
            nn.init.zeros_(self.mem_logit_gate.weight)
            nn.init.zeros_(self.mem_logit_gate.bias)
            self.mem_logit_distribution_projection = nn.Linear(hidden_dim, num_letters, bias=True)
            nn.init.kaiming_uniform_(self.mem_logit_distribution_projection.weight, a=5 ** 0.5)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mem_logit_distribution_projection.weight)
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.mem_logit_distribution_projection.bias, -bound, bound)


        self.sc_decoder = SideChainDecoder(
            self.torsion_bins, hidden_dim * 2, hidden_dim, tosion_decoder_layer, tosion_decoder_head, dropout)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        if (encode_mpnn and pre_prot_embed):
            self._initialize_pretrained_protein_encoder(mpnn_encoder_pretrain, freeze_param=freeze_encoder_param)

        if ligmpnn_init:
            self._initialize_ligmpnn_encoder_decoder(mpnn_encoder_pretrain)

        if freeze_modules:
            self._freeze_by_config(freeze_modules)

        logging.info(f"self.tm_raw == {self.tm_raw}")
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

    def _freeze_module_parameters(self, module, module_name):
        if module is None:
            return 0

        frozen_params = 0
        for _, param in module.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()

        if frozen_params > 0:
            logger.info(f'freeze parameters of {module_name}: {frozen_params:,}')
        return frozen_params

    def _freeze_by_config(self, module_names):
        # 支持两种格式:
        #   1) list  : 冻结列出的模块 (旧行为)
        #   2) dict {mode: all_except, keep: [...]} : 只训练 keep 里的顶层模块, 其余全部冻结
        if isinstance(module_names, dict):
            mode = module_names.get('mode', '')
            if mode == 'all_except':
                self._freeze_all_except(module_names.get('keep', []))
            else:
                logger.warning(f"freeze config: unknown mode '{mode}', skipping")
            return
        frozen_total = 0
        for name in module_names:
            module = getattr(self, name, None)
            if module is not None:
                frozen_total += self._freeze_module_parameters(module, name)
            else:
                logger.warning(f"freeze config: module '{name}' not found in ABACUST, skipping")
        logger.info(f'freeze_by_config total frozen params: {frozen_total:,}')

    def _freeze_all_except(self, keep_names):
        keep = set(keep_names)
        frozen_total = 0
        kept_total = 0
        for name, param in self.named_parameters():
            top = name.split('.')[0]
            if top in keep:
                kept_total += param.numel()
            else:
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_total += param.numel()
        logger.info(f'freeze_all_except keep={sorted(keep)} | frozen={frozen_total:,} trainable_kept={kept_total:,}')

    def initialize_mem_logit_gate_encoder_from_encoder(self):
        if self.mem_logit_gate_encoder_layers is None:
            return
        self.mem_logit_gate_encoder_layers.load_state_dict(self.encoder_layers.state_dict())
        for param in self.mem_logit_gate_encoder_layers.parameters():
            param.requires_grad = True
        logger.info("initialized mem_logit_gate_encoder_layers from encoder_layers")


    # def _freeze_esm_parameter(self, ):
    #     for name, param in self.esm_model.named_parameters():
    #         param.requires_grad = False
    #     logger.info(f'freeze parameters of esm model')


    def forward(
        self,
        X,
        S,
        mask,          #源自['node_mask']
        chain_M,           #源自["chainidx"]
        residue_idx,       # res的序列
        chain_encoding_all, #也是chain的掩码，但是不在一个device上
        randn,
        lig_node_attr, lig_edge_attr, lig_edge_index,
        pdbtm_regions, G, N,
        pdbname = None,
        use_input_decoding_order=False, decoding_order=None,
        edge_mem_mask = None,
        lig_mask=None, seq_mask=None,
        esm_embedding=None, last_iter_S_embed=None,
        unimol_reprs=None,
        torsion_angles=None, alt_chi_angles=None, torsion_angle_mask=None,
        prev_tokens=None,
        plip_anno_itype_list=None
        ):

        """ Graph-conditioned sequence model """
        device=X.device
        B, L = X.shape[:2]
        prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()

        # CFG: randomly drop membrane condition during training
        if self.training and self.cfg_enabled:
            if random.random() < self.cfg_dropout_prob:
                G = torch.zeros_like(G)
                N = torch.zeros_like(N)

        chain_M = chain_M*mask #update chain_M to include missing regions
        torsion_angle_mask = torsion_angle_mask * prot_mask[..., None]
        #这里的torsion_angle_mask源自chi_mask，标记了一个氨基酸哪些扭转角是有的，哪些是没有的，这个训练的时候也是需要被掩盖掉的


        #对x的坐标进行扰动
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)


        logger.info(f"Preprocessing {pdbname}.")
        ########################################################################################################################
        # 蛋白部分的编码前信息，X直接过pifold
        # 这部分信息旋转不变的，pifold里内置了旋转不变的逻辑
        # 这里的S貌似只是过了一个掩码，实际上没用上
        if self.pre_prot_embed:#true
            prot_X = X * prot_mask[..., None, None]
            pre_h_V_encoder = self.prot_encoder_pifold(prot_X, prot_mask, S)
            if self.encode_mpnn:# false
                pre_h_V_encoder_mpnn = self.prot_encoder_mpnn(prot_X, S, prot_mask, residue_idx, chain_encoding_all)
                pre_h_V_encoder = pre_h_V_encoder_mpnn + pre_h_V_encoder
        else:
            pre_h_V_encoder = 0
        ##################################################################################################################
        #处理lig的节点信息
        all_zeros_lig_mask = torch.all(lig_mask == 0, 1) #返回张量[B],标记的是每一个样本是不是所有的mask都不是配体的，即没有lig
        all_zeros_lig_unmask_bidx = torch.arange(B).long().to(device)[~all_zeros_lig_mask]
        #返回的是每一个样本的idx，如没有配体就是0

        # 跟模型权重同步
        lig_node = torch.zeros((B, L, self.hidden_dim), device=device)
        lig_X = X[:, :, 1] * lig_mask[..., None]
        #提取所有lig的坐标（每个lig的坐标是在x张量的第二个位置上?)

        if (len(all_zeros_lig_unmask_bidx) > 0):
            #如果存在一些样本的lig，则将这些配体的lig的坐标，节点特征，边索引，边特征都提取出来，
            # 之后根据配置ligencoder可以是linear或者是schnet，对其进行编码，如果嵌入了unimol的特征的话则加在一起。


            unmasked_lig_X = lig_X[all_zeros_lig_unmask_bidx]
            unmasked_lig_node_attr = lig_node_attr[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_index = lig_edge_index[all_zeros_lig_unmask_bidx]
            unmasked_lig_edge_attr = lig_edge_attr[all_zeros_lig_unmask_bidx]
            unmasked_unimol_reprs = unimol_reprs[all_zeros_lig_unmask_bidx]

            unmasked_lig_node = self.lig_encoder(unmasked_lig_node_attr, unmasked_lig_X, unmasked_lig_edge_index, unmasked_lig_edge_attr)

            #之后如果有unimol嵌入的话，将原先的ligencoder输出和激活的unimol表示相加之后乘以掩码，输回到lignode里面
            if self.embed_unimol_reprs:
                unmasked_unimol_reprs = self.unimol_repr_activate(unmasked_unimol_reprs)
                unmasked_lig_node = (unmasked_lig_node + unmasked_unimol_reprs) * lig_mask[all_zeros_lig_unmask_bidx][..., None]
            lig_node[all_zeros_lig_unmask_bidx] = unmasked_lig_node
            #lignode是一个张量，第一维度是B，b下面是unimol表征和logencoder的输出特征的求和

        #加入plip信息：对键类型的注释
        lig_node = lig_node + self.lig_plip_itype_embedding(plip_anno_itype_list.long())

        ####################################################################################################################
        # 构建边信息
        # 干两件事，第一是构造KNN图，第二是得到e： [25种原子对距离RBF + 相对序号embedding + 同链标记 + 边类型embedding（标记是prot还是lig）]
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)
        h_E = self.W_e(E)

        ###################################################################################################################
        # 整合构建节点信息
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        try:
            #加配体信息
            h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None] * lig_node + prot_mask[..., None] * pre_h_V_encoder
            # 加prot_lig位置信息
            seg_token = prot_mask * 1 + lig_mask * 2
            h_V = h_V + self.h_V_segment(seg_token.long())
        except Exception as e:
            traceback.print_exc()

        #######################################################################################
        # encoder 编码
        # mask_attend[b, i, k] = mask[b, E_idx[b, i, k]]，告诉第i个节点，其邻居的k和节点是否都是有效的
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        # 自己也要有效才行
        mask_attend = mask.unsqueeze(-1) * mask_attend
        # 真正的encoderlayer
        ###############################################################################################
        mem_logit_y_emb = None
        if self.mem_logit_modulation or self.mem_logit_gate_modulation:
            assert self.mem_logit_normal_encoder is not None, "missing logit encoder"
            result = self.mem_logit_normal_encoder(X, G, N)
            mem_logit_y_emb = result["y_emb"]
            mem_logit_y_emb = mem_logit_y_emb.to(dtype=h_V.dtype) * prot_mask[..., None]

        mem_logit_gate_h_V = None
        if self.mem_logit_gate_modulation:
            assert self.mem_logit_gate_encoder_layers is not None, "missing logit gate encoder"
            assert mem_logit_y_emb is not None, "missing logit normal encoder output"
            mem_logit_gate_h_V = h_V + mem_logit_y_emb
            mem_logit_gate_h_E = h_E
            for layer in self.mem_logit_gate_encoder_layers:
                mem_logit_gate_h_V, mem_logit_gate_h_E = layer(
                    mem_logit_gate_h_V, mem_logit_gate_h_E, E_idx, mask, mask_attend
                )
            if self.mem_logit_gate_zero_encoder:
                mem_logit_gate_h_V = torch.zeros_like(mem_logit_gate_h_V)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        # 注入膜信息用于编码（encoder之后）
        if self.mem_net is not None:
            h_V = self.mem_net(h_V, X, G, N, prot_mask)

        #################################################################################################
        # 通过mask确定解码顺序，因为是nar，所以可见。
        # prev_token不用给mask，因为abacust_mem/src/fasde/modules/sequence_designer.py输入的时候就已经mask过了，但是torsion要额外给mask
        try:
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
            # mask_bw = mask_1D * mask_attend
            mask_bw = mask_1D * (1. - mask_attend)
            mask_fw = mask_1D * (1. - mask_attend)
            sc_mask = (torch.all(torch.stack([(prev_tokens >= 4), prev_tokens!=24], -1), -1)).float()
            # 这里prevtokens是函数直接传入的
            prev_torsion = sc_mask[..., None] * torsion_angles
            prev_alt_torsion = sc_mask[..., None] * torsion_angles
            prev_torsion_mask = sc_mask[..., None] * torsion_angle_mask

        except Exception as e:
            import traceback; traceback.print_exc();

        ##############################################################################################################
        #融入序列信息，包括embedding还有esm
        h_S = self.W_s(prev_tokens) + self.W_esm_act(self.W_esm_LN(last_iter_S_embed))
        #last_iter_S_embed在第一次forward函数的时候是0，第二次的时候部分位置是上一轮的esm向前传播结果，具体哪些位置取决于curtimestep

        #################################################################################################################
        # 构建侧链主导的节点和边，这里提供x是方便根据chi角还原侧链位置
        sc_mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        sc_mask_attend = mask.unsqueeze(-1) * sc_mask_attend
        h_C, h_EC_ = self.W_c(prev_torsion, prev_torsion_mask, E_idx, aatype=prev_tokens,
            mask_attend=sc_mask_attend, backbone_coords=X, mask=mask, lig_mask=lig_mask, mask_scatom=True) # hidden_dim, 2 * hidden_dim


        ##################################################################################################################################
        # 构建序列层面的节点和边表示
        h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1)) #B,L,D
        h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES# 这里加入E是因为边特征含有序列差，链等信息


        if (len(self.aux_enc_sc) > 0):
            h_ESC = self.activate_W_sc_merge_edge(h_ESC)
            #一个线性连接层
            scEnc_h_V = h_SC + h_V
            scEnc_h_E = h_ESC + h_E

            if (self.aux_enc_sc == 'PiFold'):
                #目前是PIfold模式
                mask_pair = (mask[:, :, None] * mask[:,None,:])
                edge_mask = gather_edges(mask_pair[:,:,:,None], E_idx)[:,:,:,0].bool()
                # B,L,K ，Eidx就是B,L,K
                # 负责再传播一遍序列和侧链信息
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
            h_ESC = torch.cat([h_E, h_ESC_update], -1)
            ############################################
            # 将原始的HV信息和侧链的更新节点信息轮番汇总到每一边上
            # cat_neirghbor_nodes本身是将节点concat到边上的函数，输出是边
            h_EX_encoder = cat_neighbors_nodes(h_SC_update, h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) # 这个变量带有h_SC_update，h_E，h_V的concat，之后每一轮decoder都会反复注入原始信息，保证不会忘掉原始输入

        else:
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # Build encoder embeddings
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # Compute depth conditioning for AdaLN
        adaln_depth_cond = None
        if self.depth_inject_method != 'none':
            adaln_depth_cond, _, _ = self.depth_encoder(X, G, N)

        # 膜信息(深度+角度)过基础MLP -> mem_edge_dim, 广播到每条边并拼到 h_ESV 后面
        mem_edge = None
        if self.mem_edge_encoder is not None:
            mem_node = self.mem_edge_encoder(X, G, N, prot_mask)
            mem_edge = mem_node.unsqueeze(2).expand(-1, -1, E_idx.shape[-1], -1)

        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx) # 2 * hidden
            h_ESV = mask_bw * (h_ESCV) + h_EXV_encoder_fw
            if mem_edge is not None:
                h_ESV = torch.cat([h_ESV, mem_edge], -1)
            if self.depth_inject_method != 'none':
                h_V = layer(h_V, h_ESV, adaln_depth_cond, mask)
            else:
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)

        if self.mem_logit_modulation:
            assert self.mem_logit_bias is not None, "missing logit bias"
            logits = logits + self.mem_logit_bias(torch.cat([h_V, mem_logit_y_emb], dim=-1))

        gate = None
        log_probs = F.log_softmax(logits, dim=-1)

        if self.mem_logit_gate_modulation:
            assert self.mem_logit_gate is not None, "missing logit gate"
            assert self.mem_logit_distribution_projection is not None, "missing membrane distribution projection"
            gate = torch.sigmoid(self.mem_logit_gate(torch.cat([mem_logit_gate_h_V, mem_logit_y_emb], dim=-1)))
            membrane_logits = self.mem_logit_distribution_projection(mem_logit_y_emb)
            logits = logits + gate * membrane_logits
            log_probs = F.log_softmax(logits, dim=-1)

        #########################################################
        # self.region_decoder = RegionDecoder(d_in=self.hidden_dim *4)

        # regoin_logits = self.region_out(h_V)
        # region_log_probs = F.log_softmax(regoin_logits,dim = -1)


        ##############################################################


        ## teacher forcing training p(sc^i | aa^i, aa^j, sc^j) * p(aa^i | aa^j, sc^j) = p(aa^i, sc^i | aa^j, sc^j)
        all_h_S = self.W_s(S)
        sc_res_ctx = torch.cat([all_h_S, h_V], -1)
        # 获取侧链的环境，包括先前的节点信息，以及与之想对应的序列的嵌入信息
        sc_logits = self.sc_decoder(sc_res_ctx, torsion_angles, torsion_angle_mask)
        log_sc_probs = F.log_softmax(sc_logits, dim=-1)

        if torch.isnan(log_probs).any():
            raise RuntimeError("log_probs contains NaN. Exiting due to numerical instability.")

        return log_probs, prev_tokens, log_sc_probs, prev_torsion_mask, pdbtm_regions, edge_mem_mask, gate  # gate for tb logging


    def nar_sample(self,
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                lig_node_attr, lig_edge_attr, lig_edge_index,
                pdbtm_regions,
                G,
                N,
                lig_mask, iter_num=5, temperature=1.0, seq_mask_pocket=None,
                backbone_affine_tensor=None, backbone_angles_sin_cos=None,
                esm_embedding=None,
                unimol_reprs=None,
                initial_S=None,  #新的
                esm_batcher=None, esm_model=None, esm_alphabet=None,
                mask_mode='aatype_nll',
                return_pdb_ctx_iter=-1, #新的
                packing_only=False, noise_scale=1.0,
                start_from_init=False,
                bg_dist=None,
                bg_weight=0.5,
                fixed_positions_mask=None,
                plip_anno_itype_list=None,
                cfg_guidance_scale=1.0,
                mem_bg_weight=0.0,
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
        if DEBUG:
            import pdb;pdb.set_trace()

        if self.pre_prot_embed:
            # Prepare node and edge embeddings
            prot_X = X * prot_mask[..., None, None]
            pre_h_V_encoder = self.prot_encoder_pifold(prot_X, prot_mask, S)
            if self.encode_mpnn:
                pre_h_V_encoder_mpnn = self.prot_encoder_mpnn(prot_X, S, prot_mask, residue_idx, chain_encoding_all)
                pre_h_V_encoder = pre_h_V_encoder_mpnn + pre_h_V_encoder
        else:
            pre_h_V_encoder = 0

        if DEBUG:
            import pdb;pdb.set_trace()

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

        try:
            lig_node = lig_node + self.lig_plip_itype_embedding(plip_anno_itype_list.long())
        except Exception as e:
            import traceback;traceback.print_exc()
            import pdb;pdb.set_trace()
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        # h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None] * lig_node + prot_mask[..., None] * pre_h_V_encoder

        try:
            h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None] * lig_node + prot_mask[..., None] * pre_h_V_encoder
            # # import pdb;pdb.set_trace()
            # 将蛋白质节点信息和lig节点信息进行融合
        except Exception as e:
            traceback.print_exc()
            import pdb;pdb.set_trace()
        ###########################################################################################################################

        seg_token = prot_mask * 1 + lig_mask * 2
        h_V_seg = self.h_V_segment(seg_token.long())
        h_V = h_V + h_V_seg
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        _mem_logit_y_emb = None
        if (self.mem_logit_modulation or self.mem_logit_gate_modulation) and G is not None and N is not None:
            result = self.mem_logit_normal_encoder(X, G, N)
            _mem_logit_y_emb = result["y_emb"]
            _mem_logit_y_emb = _mem_logit_y_emb.to(dtype=h_V.dtype) * prot_mask[..., None]

        mem_logit_gate_h_V_init = None
        if self.mem_logit_gate_modulation:
            assert self.mem_logit_gate_encoder_layers is not None, "missing logit gate encoder"
            assert _mem_logit_y_emb is not None, "missing logit normal encoder output"
            mem_logit_gate_h_V_init = h_V + _mem_logit_y_emb
            mem_logit_gate_h_E_init = h_E
            for layer in self.mem_logit_gate_encoder_layers:
                mem_logit_gate_h_V_init, mem_logit_gate_h_E_init = layer(
                    mem_logit_gate_h_V_init, mem_logit_gate_h_E_init, E_idx, mask, mask_attend
                )
            if self.mem_logit_gate_zero_encoder:
                mem_logit_gate_h_V_init = torch.zeros_like(mem_logit_gate_h_V_init)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        # CFG: save pre-membrane encoder output for unconditional path
        _cfg_active = cfg_guidance_scale is not None and cfg_guidance_scale > 1.0
        _cfg_h_V_pre_mem = h_V.clone() if _cfg_active else None

        # 注入膜信息用于编码（encoder之后）
        if self.mem_net is not None:
            h_V = self.mem_net(h_V, X, G, N, prot_mask)

        # CFG: compute unconditional encoder output for guidance
        h_V_init_uncond = None
        _cfg_adaln_uncond = None
        if _cfg_active:
            _G_zero = torch.zeros_like(G)
            _N_zero = torch.zeros_like(N)
            if self.mem_net is not None:
                h_V_init_uncond = self.mem_net(_cfg_h_V_pre_mem, X, _G_zero, _N_zero, prot_mask)
            else:
                h_V_init_uncond = _cfg_h_V_pre_mem
            if self.depth_inject_method != 'none':
                _cfg_adaln_uncond, _, _ = self.depth_encoder(X, _G_zero, _N_zero)

        h_V_init = h_V
        h_E_init = h_E
        cur_iter_masked_S = initial_S
        init_iter_mask = cur_iter_masked_S == 0
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
        # Pre-compute mem_logit_y_emb (structure-only, no need to recompute each iter)
        _debug_mem_logit_dump = os.environ.get("ABACUST_MEM_DEBUG_DUMP", "0") == "1"
        _mem_logit_debug = [] if _debug_mem_logit_dump else None
        for cur_iter_num in range(iter_num):
            cur_iter_mask = cur_iter_masked_S == 0

            if (cur_iter_num > 0):
                input_esm_tokens, raw_esm_tokens = converter_from_af2_to_esm(cur_iter_S, prot_mask)
                try:
                    with torch.no_grad():
                        last_iter_esm_embed = torch.zeros((B, L, 1280)).to(device)
                        output = esm_model(input_esm_tokens, repr_layers=[33], return_contacts=False)
                        esm_rep_ = output['representations'][33][:, 1:-1].detach()
                        last_iter_S_embed = torch.where((prot_mask == 1)[..., None], esm_rep_, last_iter_esm_embed)
                except Exception as e:
                    traceback.print_exc()
                    import pdb;pdb.set_trace()
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
                _cfg_h_ESC_merged = h_ESC.clone() if _cfg_active else None  # CFG: save for uncond path
                scEnc_h_V = h_SC + h_V
                scEnc_h_E = h_ESC + h_E
                # scEnc_h_V = self.activate_scEnc_h_V(torch.cat([h_SC, h_V], -1))
                # scEnc_h_E = self.activate_scEnc_h_E(cat_neighbors_nodes(scEnc_h_V, torch.cat([h_ESC, h_E], -1), E_idx))
                if (self.aux_enc_sc == 'PiFold'):
                    #目前是PIfold模式
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
                #将h_SC_update，h_E，h_V融合起来，

            else:
                h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
                h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            # Compute depth conditioning for AdaLN (per iteration, reuses same X/G/N)
            adaln_depth_cond = None
            if self.depth_inject_method != 'none':
                adaln_depth_cond, _, _ = self.depth_encoder(X, G, N)

            # 膜信息(深度+角度) -> mem_edge_dim, 广播到边并拼到 h_ESV 后面
            mem_edge = None
            if self.mem_edge_encoder is not None:
                mem_node = self.mem_edge_encoder(X, G, N, prot_mask)
                mem_edge = mem_node.unsqueeze(2).expand(-1, -1, E_idx.shape[-1], -1)

            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx)
                h_ESV = mask_bw * h_ESCV + h_EXV_encoder_fw
                if mem_edge is not None:
                    h_ESV = torch.cat([h_ESV, mem_edge], -1)
                if self.depth_inject_method != 'none':
                    h_V = layer(h_V, h_ESV, adaln_depth_cond, mask)
                else:
                    h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)

            # ===== CFG: compute unconditional logits and apply guidance =====
            if _cfg_active:
                logits_cond = logits
                # Unconditional path: rerun from h_V_init_uncond
                _h_V_u = h_V_init_uncond.clone()
                if (len(self.aux_enc_sc) > 0):
                    _scEnc_h_V_u = h_SC + _h_V_u
                    _scEnc_h_E_u = _cfg_h_ESC_merged + h_E
                    if (self.aux_enc_sc == 'PiFold'):
                        _mp = (mask[:, :, None] * mask[:, None, :])
                        _em = gather_edges(_mp[:,:,:,None], E_idx)[:,:,:,0].bool()
                        _h_SC_u, _h_ESC_u = self.aux_scEnc(_scEnc_h_V_u, _scEnc_h_E_u, E_idx, mask, _em)
                    elif (self.aux_enc_sc == 'MPNN'):
                        _ma = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
                        _ma = mask.unsqueeze(-1) * _ma
                        _sv, _se = _scEnc_h_V_u, _scEnc_h_E_u
                        for _lyr in self.aux_scEnc:
                            _sv, _se = _lyr(_sv, _se, E_idx, mask, _ma)
                        _h_SC_u, _h_ESC_u = _sv, _se
                    _h_V_u = _h_SC_u + _h_V_u
                    _h_ESC_u = torch.cat([h_E, _h_ESC_u], -1)
                    _h_EX_u = cat_neighbors_nodes(_h_SC_u, h_E, E_idx)
                    _h_EXV_u = cat_neighbors_nodes(_h_V_u, _h_EX_u, E_idx)
                else:
                    _h_EX_u = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
                    _h_EXV_u = cat_neighbors_nodes(_h_V_u, _h_EX_u, E_idx)
                    _h_ESC_u = h_ESC
                _h_EXV_fw_u = mask_fw * _h_EXV_u
                for _lyr in self.decoder_layers:
                    _h_ESCV_u = cat_neighbors_nodes(_h_V_u, _h_ESC_u, E_idx)
                    _h_ESV_u = mask_bw * _h_ESCV_u + _h_EXV_fw_u
                    if mem_edge is not None:
                        _h_ESV_u = torch.cat([_h_ESV_u, mem_edge], -1)
                    if self.depth_inject_method != 'none':
                        _h_V_u = _lyr(_h_V_u, _h_ESV_u, _cfg_adaln_uncond, mask)
                    else:
                        _h_V_u = _lyr(_h_V_u, _h_ESV_u, mask)
                logits_uncond = self.W_out(_h_V_u)
                logits = logits_uncond + cfg_guidance_scale * (logits_cond - logits_uncond)
            # ===== End CFG =====

            probs = F.softmax(logits, dim=-1)

            # mem_logit_modulation: additive bias on logits
            if self.mem_logit_modulation and _mem_logit_y_emb is not None:
                logits = logits + self.mem_logit_bias(torch.cat([h_V, _mem_logit_y_emb], dim=-1))
                probs = F.softmax(logits, dim=-1)

            _mem_logit_debug_item = None
            _probs_pre_gate = probs.detach().cpu() if _debug_mem_logit_dump else None
            # mem_logit_gate_modulation: gate-scaled membrane logits
            if self.mem_logit_gate_modulation and _mem_logit_y_emb is not None:
                _gate = torch.sigmoid(self.mem_logit_gate(torch.cat([mem_logit_gate_h_V_init, _mem_logit_y_emb], dim=-1)))
                _membrane_logits = self.mem_logit_distribution_projection(_mem_logit_y_emb)
                logits = logits + _gate * _membrane_logits
                probs = F.softmax(logits, dim=-1)
                if _debug_mem_logit_dump:
                    _membrane_distribution = F.softmax(_membrane_logits, dim=-1)
                    _mem_logit_debug_item = {
                        "iter": int(cur_iter_num),
                        "gate": _gate.detach().cpu(),
                        "membrane_distribution": _membrane_distribution.detach().cpu(),
                        "probs_pre_gate": _probs_pre_gate,
                        "probs_post_gate": probs.detach().cpu(),
                    }

            # Membrane-depth-based background reweighting
            if mem_bg_weight > 0 and G is not None and N is not None and N.norm(dim=-1).max() > 1e-3:
                gr = G[:, :, :3]
                gt = G[:, :, 3:]
                ca = X[:, :, 1, :]
                r_mem = torch.bmm(gr, (ca + gt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
                half_thickness = N.norm(dim=-1).clamp(min=1e-6)
                unit_normal = N / half_thickness.unsqueeze(-1)
                signed_depth = (r_mem * unit_normal.unsqueeze(1)).sum(dim=-1)
                abs_depth = signed_depth.abs()
                interface_offset = abs_depth - half_thickness.unsqueeze(1)
                gauss_weight = torch.exp(-((interface_offset - (-5.0)) ** 2) / (2.0 * 2.0 ** 2))
                per_residue_weight = mem_bg_weight * gauss_weight
                bg_wy = torch.zeros_like(probs)
                bg_wy[:, :, 21] = 0.55   # W
                bg_wy[:, :, 22] = 0.35   # Y
                bg_wy[:, :, 14] = -0.05  # L (suppress)
                probs = (1 - per_residue_weight.unsqueeze(-1)) * probs + per_residue_weight.unsqueeze(-1) * bg_wy
                probs = probs.clamp(min=0)
                logits = torch.log(probs + 1e-8)

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
            #     # all_refined_seq, refined_tokens, raw_cur_nll = msa_refine(seq_list)
            #     refined_tokens = refined_tokens.to(device)

            #     cur_iter_S[:, :prot_aa_num] = refined_tokens
            #     # import pdb; pdb.set_trace()
            #     output_scores = (1 - init_iter_mask.float()) * 2# + refined_raw_cur_nll * init_iter_mask.float()
            #     output_scores[:, :prot_aa_num] = refined_raw_cur_nll.to(device)
            # else:
            raw_cur_nll = torch.log(torch.gather(probs, -1, cur_iter_S[..., None])[..., 0])
            output_scores = (1 - init_iter_mask.float()) * 2 + raw_cur_nll * init_iter_mask.float()
            if _debug_mem_logit_dump and _mem_logit_debug_item is not None:
                _mem_logit_debug_item["probs_final"] = probs.detach().cpu()
                _mem_logit_debug_item["sampled_tokens"] = raw_S_iter.detach().cpu()
                _mem_logit_debug_item["raw_cur_nll"] = raw_cur_nll.detach().cpu()
                _mem_logit_debug_item["output_scores"] = output_scores.detach().cpu()
                _mem_logit_debug_item["cur_iter_mask"] = cur_iter_mask.detach().cpu()
                _mem_logit_debug_item["cur_iter_masked_S"] = cur_iter_masked_S.detach().cpu()
                _mem_logit_debug.append(_mem_logit_debug_item)

            if start_from_init:
                if (cur_iter_num == 0):
                    cur_iter_S = S.long()

            if fixed_positions_mask is not None:
                try:
                    cur_iter_S = torch.where(fixed_positions_mask.bool(), S.long(), cur_iter_S.long())
                except Exception as e:
                    fixed_positions_mask = torch.from_numpy(fixed_positions_mask).bool().to(S.device)
                    # 然后再用 torch.where
                    cur_iter_S = torch.where(fixed_positions_mask, S.long(), cur_iter_S.long())#如果输入时np数组的话，先将其转换成为张量之后再进行布尔转换

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

            elif (mask_mode == 'aatype_nll'):
                output_scores_mask = init_iter_mask
                cur_iter_p = 1 - (cur_iter_num + 1) / iter_num
                iter_output_scores = output_scores
                iter_output_scores = iter_output_scores * (cur_iter_masked_S == 0).float() + 2 * (cur_iter_masked_S != 0).float()
                skeptical_mask_prot = _skeptical_unmasking(iter_output_scores[:,:prot_len], output_scores_mask[:, :prot_len], cur_iter_p)
                cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot, 0.0)
                cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)


            # output_scores_mask = init_iter_mask
            # cur_iter_p = 1 - (cur_iter_num + 1) / iter_num
            # iter_output_scores = output_scores
            # iter_output_scores = iter_output_scores * (cur_iter_masked_S == 0).float() + 2 * (cur_iter_masked_S != 0).float()
            # skeptical_mask_prot = _skeptical_unmasking(iter_output_scores[:,:prot_len], output_scores_mask[:, :prot_len], cur_iter_p)
            # cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot, 0.0)
            # cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)

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
            'protein_mask': prot_mask.cpu(),
            # 'torsion_discrete': iter_torsion_discrete_list.cpu(),
            'torsion_continuous': iter_torsion_list.cpu(),
            # "mem_position":mem_position[...,:prot_len] #标记蛋白质在膜中的相对深度，
        }
        if _debug_mem_logit_dump:
            design_ctx['mem_logit_debug'] = _mem_logit_debug
        # print(design_ctx)
        logger.info("nar_sample finished")
        return design_ctx





    # def nar_sample_selfcond(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, #这个函数还没更新

    #     lig_node_attr, lig_edge_attr, lig_edge_index,
    #     lig_mask, iter_num=5, temperature=1.0, seq_mask_pocket=None,
    #     backbone_affine_tensor=None, backbone_angles_sin_cos=None,
    #     esm_embedding=None, unimol_reprs=None,
    #     esm_batcher=None, esm_model=None, esm_alphabet=None, mask_mode='aatype_nll',
    #     packing_only=False, noise_scale=0.0,
    #     start_from_init=False, bg_dist=None, bg_weight=0.5, fixed_positions_mask=None):

    #     device=X.device
    #     B, L = X.shape[:2]
    #     esm_embedding = None
    #     res_num = X.shape[-1]
    #     bsz = X.shape[0]
    #     prot_mask = (torch.all(torch.stack([(1-lig_mask), mask], -1), -1)).float()
    #     prot_len = prot_mask.sum(-1)[0].long().item()
    #     chain_M = chain_M*mask #update chain_M to include missing regions
    #     # import pdb; pdb.set_trace()
    #     if self.augment_eps > 0:
    #         X = X + self.augment_eps * torch.randn_like(X) * noise_scale

    #     initial_S_ = S.clone().float()
    #     initial_S_pocket = initial_S_ * (1 - seq_mask_pocket.float()) + lig_mask * 24 + (1 - mask)
    #     initial_S = initial_S_ * (1 - prot_mask)

    #     if self.pre_prot_embed:
    #         # Prepare node and edge embeddings
    #         prot_X = X * prot_mask[..., None, None]
    #         pre_h_V_encoder = self.prot_encoder_pifold(prot_X, prot_mask, S)
    #         if self.encode_mpnn:
    #             pre_h_V_encoder_mpnn = self.prot_encoder_mpnn(prot_X, S, prot_mask, residue_idx, chain_encoding_all)
    #             pre_h_V_encoder = pre_h_V_encoder_mpnn + pre_h_V_encoder
    #     else:
    #         pre_h_V_encoder = 0

    #     all_zeros_lig_mask = torch.all(lig_mask == 0, 1)
    #     all_zeros_lig_unmask_bidx = torch.arange(B).long().to(device)[~all_zeros_lig_mask]
    #     lig_node = torch.zeros((B, L, self.hidden_dim), device=device)
    #     lig_X = X[:, :, 1] * lig_mask[..., None]
    #     # import pdb; pdb.set_trace()
    #     if (len(all_zeros_lig_unmask_bidx) > 0):
    #         unmasked_lig_X = lig_X[all_zeros_lig_unmask_bidx]
    #         unmasked_lig_node_attr = lig_node_attr[all_zeros_lig_unmask_bidx]
    #         unmasked_lig_edge_index = lig_edge_index[all_zeros_lig_unmask_bidx]
    #         unmasked_lig_edge_attr = lig_edge_attr[all_zeros_lig_unmask_bidx]
    #         unmasked_unimol_reprs = unimol_reprs[all_zeros_lig_unmask_bidx]

    #         unmasked_lig_node = self.lig_encoder(unmasked_lig_node_attr, unmasked_lig_X, unmasked_lig_edge_index, unmasked_lig_edge_attr)
    #         if self.embed_unimol_reprs:
    #             unmasked_unimol_reprs = self.unimol_repr_activate(unmasked_unimol_reprs)
    #             unmasked_lig_node = (unmasked_lig_node + unmasked_unimol_reprs) * lig_mask[all_zeros_lig_unmask_bidx][..., None]
    #         lig_node[all_zeros_lig_unmask_bidx] = unmasked_lig_node

    #     # Prepare node and edge embeddings
    #     E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, lig_mask=lig_mask)

    #     h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    #     h_V = h_V*(1-lig_mask[...,None]) + lig_mask[...,None]*lig_node + prot_mask[..., None] * pre_h_V_encoder
    #     seg_token = prot_mask * 1 + lig_mask * 2
    #     h_V_seg = self.h_V_segment(seg_token.long())
    #     h_V = h_V + h_V_seg
    #     h_E = self.W_e(E)

    #     # Encoder is unmasked self-attention
    #     mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
    #     mask_attend = mask.unsqueeze(-1) * mask_attend
    #     for layer in self.encoder_layers:
    #         h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    #     h_V_init = h_V
    #     h_E_init = h_E
    #     cur_iter_masked_S = initial_S
    #     init_iter_mask = cur_iter_masked_S == 0
    #     init_iter_mask_pocket = initial_S_pocket == 0

    #     cur_iter_S = None
    #     iter_seq_list = []
    #     iter_ll_list = []
    #     iter_identity_list = []
    #     iter_identity_list_pocket = []
    #     iter_torsion_list = []
    #     iter_torsion_discrete_list = []
    #     # given pre-designed torsion information in cur_iter_torsion
    #     cur_iter_torsion = torch.zeros((B, L, 4)).float().to(device)
    #     cur_iter_torsion_mask = torch.zeros((B, L, 4)).float().to(device)
    #     # if side-chain packing only, then cur_iter_torsion is None
    #     for cur_iter_num in range(iter_num):
    #         cur_iter_mask = cur_iter_masked_S == 0
    #         if (cur_iter_num > 0):
    #             input_esm_tokens = converter_from_af2_to_esm(cur_iter_S, prot_mask)

    #             with torch.no_grad():
    #                 last_iter_esm_embed = torch.zeros((B, L, self.esm_dim)).to(device)
    #                 output = esm_model(input_esm_tokens, repr_layers=[self.rep_layer], return_contacts=False)
    #                 esm_rep_ = output['representations'][self.rep_layer][:, 1:-1].detach()
    #                 last_iter_S_embed = torch.where((prot_mask == 1)[..., None], esm_rep_, last_iter_esm_embed)

    #         else:
    #             last_iter_S_embed = torch.zeros((B, L, self.esm_dim)).to(device)

    #         # Concatenate sequence embeddings for autoregressive decoder
    #         h_S = self.W_s(cur_iter_masked_S.long()) + self.W_esm(last_iter_S_embed)

    #         sc_mask = (torch.all(torch.stack([(cur_iter_masked_S >= 4), cur_iter_masked_S!=24], -1), -1)).float()
    #         cur_iter_torsion = sc_mask[..., None] * cur_iter_torsion
    #         cur_iter_torsion_mask = sc_mask[..., None] * cur_iter_torsion_mask
    #         sc_mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
    #         sc_mask_attend = mask.unsqueeze(-1) * sc_mask_attend

    #         h_C, h_EC_ = self.W_c(cur_iter_torsion, cur_iter_torsion_mask, E_idx, aatype=cur_iter_masked_S.long(),
    #             mask_attend=sc_mask_attend, backbone_coords=X, mask=mask, mask_scatom=True)
    #         h_SC = self.activate_W_sc_node(torch.cat([h_S, h_C], -1))
    #         h_ESC = cat_neighbors_nodes(h_SC, h_E, E_idx) + self.activate_W_sc_edge(h_EC_) # + pre_h_ES

    #         mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    #         mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
    #         # mask_bw = mask_1D * mask_attend
    #         mask_bw = mask_1D * (1. - mask_attend)
    #         mask_fw = mask_1D * (1. - mask_attend)

    #         # Build encoder embeddings
    #         if (len(self.aux_enc_sc) > 0):
    #             h_ESC = self.activate_W_sc_merge_edge(h_ESC)
    #             scEnc_h_V = h_SC + h_V
    #             scEnc_h_E = h_ESC + h_E
    #             if (self.aux_enc_sc == 'PiFold'):
    #                 mask_pair = (mask[:, :, None] * mask[:,None,:])
    #                 edge_mask = gather_edges(mask_pair[:,:,:,None], E_idx)[:,:,:,0].bool()
    #                 h_SC_update, h_ESC_update = self.aux_scEnc(scEnc_h_V, scEnc_h_E, E_idx, mask, edge_mask) # hidden_dim

    #             elif (self.aux_enc_sc == 'MPNN'):
    #                 mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
    #                 mask_attend = mask.unsqueeze(-1) * mask_attend
    #                 for layer in self.aux_scEnc:
    #                     scEnc_h_V, scEnc_h_E = layer(scEnc_h_V, scEnc_h_E, E_idx, mask, mask_attend)
    #                 h_SC_update = scEnc_h_V
    #                 h_ESC_update = scEnc_h_E

    #             h_ESC = torch.cat([h_E, h_ESC_update], -1)
    #             h_EX_encoder = cat_neighbors_nodes(h_SC_update, h_E, E_idx)
    #             h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) # 2 * hidden

    #         else:
    #             h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_SC), h_E, E_idx)
    #             h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    #         h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    #         for layer in self.decoder_layers:
    #             # Masked positions attend to encoder information, unmasked see.
    #             h_ESCV = cat_neighbors_nodes(h_V, h_ESC, E_idx)
    #             h_ESV = mask_bw * h_ESCV + h_EXV_encoder_fw
    #             h_V = layer(h_V, h_ESV, mask)

    #         logits = self.W_out(h_V)
    #         probs = F.softmax(logits, dim=-1)

    #         if bg_dist is not None and bg_weight > 0:
    #             probs = (1 - bg_weight) * probs + bg_weight * bg_dist
    #             logits = torch.log(probs + 1e-8)

    #         raw_S_iter, raw_S_iter_score = sample_from_categorical(logits, temperature)

    #         cur_iter_S = (cur_iter_mask.float() * raw_S_iter + cur_iter_masked_S).long()
    #         cur_iter_S = cur_iter_S.masked_fill((1-prot_mask).bool(), 24.0)
    #         raw_cur_ll = torch.log(torch.gather(probs, -1, cur_iter_S[..., None])[..., 0])
    #         output_scores = (1 - init_iter_mask.float()) * 2 + raw_cur_ll * init_iter_mask.float()

    #         if start_from_init:
    #             if cur_iter_num == 0:
    #                 cur_iter_S = S.long()

    #         if fixed_positions_mask is not None:
    #             # import pdb; pdb.set_trace()
    #             cur_iter_S = torch.where(fixed_positions_mask.bool(), S.long(), cur_iter_S.long())

    #         ## teacher forcing training p(sc^i | aa^i, aa^j, sc^j) * p(aa^i | aa^j, sc^j) = p(aa^i, sc^i | aa^j, sc^j)
    #         if packing_only:
    #             cur_iter_S = S.long()

    #         all_h_S = self.W_s(cur_iter_S)
    #         sc_res_ctx = torch.cat([all_h_S, h_V], -1)
    #         # import pdb; pdb.set_trace()
    #         cur_iter_torsion_mask = make_torsion_mask_from_aatype(cur_iter_S) # cur_iter_S

    #         cur_iter_disc_torsion, cur_iter_disc_torsion_ll = self.sc_decoder.sample(sc_res_ctx, cur_iter_torsion_mask, temperature)
    #         cur_iter_torsion = self.W_c.disc_to_conti.to(device)[cur_iter_disc_torsion]

    #         iter_seq_list.append(cur_iter_S)
    #         iter_ll_list.append(raw_cur_ll[:, :prot_len].mean(-1))
    #         iter_torsion_list.append(cur_iter_torsion)
    #         iter_torsion_discrete_list.append(cur_iter_disc_torsion)
    #         cur_batch_iter_ident = ((cur_iter_S == S).float() * init_iter_mask.float()).sum(-1)/(init_iter_mask.float() + 1e-10).sum(-1)
    #         iter_identity_list.append(cur_batch_iter_ident) # all identity
    #         cur_batch_iter_ident_pocket = ((cur_iter_S == S).float() * init_iter_mask_pocket.float()).sum(-1)/(init_iter_mask_pocket.float() + 1e-10).sum(-1)
    #         iter_identity_list_pocket.append(cur_batch_iter_ident_pocket) # pocket identity

    #         if (mask_mode == 'violation'):
    #             skeptical_mask_prot = get_violation_mask_from_design(cur_iter_S[:, :prot_len], cur_iter_torsion[:, :prot_len], backbone_angles_sin_cos[:, :prot_len], backbone_affine_tensor[:, :prot_len])
    #             cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot.to(device), 0.0)
    #             cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)

    #         elif (mask_mode == 'aatype_nll'):
    #             output_scores_mask = init_iter_mask
    #             cur_iter_p = 1 - (cur_iter_num + 1) / iter_num
    #             iter_output_scores = output_scores
    #             skeptical_mask_prot = _skeptical_unmasking(iter_output_scores[:,:prot_len], output_scores_mask[:, :prot_len], cur_iter_p)
    #             cur_iter_masked_S = cur_iter_S[:, :prot_len].masked_fill(skeptical_mask_prot, 0.0)
    #             cur_iter_masked_S = F.pad(cur_iter_masked_S, (0, L-prot_len, 0, 0), 'constant', 24)

    #         h_V = h_V_init
    #         h_E = h_E_init

    #     iter_seq_list = torch.stack(iter_seq_list)
    #     iter_ll_list = torch.stack(iter_ll_list)
    #     iter_identity_list = torch.stack(iter_identity_list)
    #     iter_identity_list_pocket = torch.stack(iter_identity_list_pocket)
    #     iter_torsion_discrete_list = torch.stack(iter_torsion_discrete_list)
    #     iter_torsion_list = torch.stack(iter_torsion_list)

    #     design_ctx = {
    #         'aatype': iter_seq_list.cpu(),
    #         'all_identity': iter_identity_list.cpu(),
    #         'pocket_identity': iter_identity_list_pocket.cpu(),
    #         'protein_mask': prot_mask,
    #         'torsion_discrete': iter_torsion_discrete_list,
    #         'torsion_continuous': iter_torsion_list
    #     }

    #     return design_ctx
