import os
import sys
import subprocess
import argparse
import json
import math
from pathlib import Path
from copy import deepcopy
import tempfile
import ml_collections as mlc

import numpy as np
import torch
import torch.nn.functional as F
sys.path.append("/home/chenty/abacust_mem/src/")

from rdkit import Chem
from fasde.models.diff_full_atom import DiffFullAtom
from fasde.modules.design_utils import  make_pdb_ctx_from_design

from fasde.data.utils.diffusion_utils_atoms import make_complex_feature, make_ligand_node_feature
from fasde.data.utils import features,target_features
from data_utils import pad_and_stack
from fasde.utils.relax.assess_violation import get_violation_metrics
from fasde.modules.alphafold.common.protein import from_pdb_string
import traceback

# from pytmalign import TMalign, parse_matrixfile
# tmaligner = TMalign()

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



esm_dict = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
    'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V',
]

region_dict = {
    '1': 1,
    '2': 2,
    'H': 3,
    'B': 4,
    'C': 5,
    'L': 6,
    'I': 7,
    'U': 8,
    '0': 0,#这个是录入数据的时候填充的，在将xml转换成json的时候，region中没有但是序列里有的aa被标记为0
    'F': 9
    #先这么记着，pdbtm里没给f啥意思，但是xml里面有（1ar1)之后抽空查一下怎么回事。。。
}
reverse_region_dict = {v: k for k, v in region_dict.items()}

# 1, for side one
# H|B|C|I|L, for membrane embedded
# H for alpha helix,
# B for beta strand,
# C for coiled structure
# L for membrane embedded region not crossing the membrane (loop)
# I for membrane embedded region not interacting with lipids, polypeptid segment inside a beta barrel) :
# 2 for side two :
# U for unknown / missing residue


raw_restypes = restypes
raw_restype_order = {restype: i for i, restype in enumerate(restypes)}
raw_res_id_to_aatype = {v: k for k, v in raw_restype_order.items()}

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


def base_architecture(args):
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 256)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 128)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.edge_attn_distance_cutoff = getattr(args, "edge_attn_distance_cutoff", -1)
    args.edge_embed_dim = getattr(args, "edge_embed_dim", 0)
    args.embed_edge_vectors = getattr(args, "embed_edge_vectors", False)
    args.embed_features_in_global_frame = getattr(args, "embed_features_in_global_frame", False)
    args.embed_features_in_local_frame = getattr(args, "embed_features_in_local_frame", True)
    args.embed_gvp_in_global_frame = getattr(args, "embed_gvp_in_global_frame", False)
    args.embed_gvp_in_local_frame = getattr(args, "embed_gvp_in_local_frame", True)
    args.embed_ingraham_features = getattr(args, "embed_ingraham_features", True)
    args.embed_patch_layers = getattr(args, "embed_patch_layers", 0)
    args.embed_rotation_frames = getattr(args, "embed_rotation_frames", False)
    args.embed_rotation_quaternions = getattr(args, "embed_rotation_quaternions", False)
    args.embed_scores = getattr(args, "embed_scores", True)
    args.empty_cache_freq = getattr(args, "empty_cache_freq", 0)


    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_edge_attn_layers = getattr(args, "encoder_edge_attn_layers", 0)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.equivariant_attn_layers = getattr(args, "equivariant_attn_layers", 0)
    args.et_gvp_as_feedforward = getattr(args, "et_gvp_as_feedforward", False)


    args.gvp_attention_heads = getattr(args, "gvp_attention_heads", 0)
    args.gvp_conditioning_encoder = getattr(args, "gvp_conditioning_encoder", True)
    args.gvp_conditioning_score_num_rbf = getattr(args, "gvp_conditioning_score_num_rbf", 16)
    args.gvp_conv_no_scalar_activation = getattr(args, "gvp_conv_no_scalar_activation", False)
    args.gvp_conv_no_vector_activation = getattr(args, "gvp_conv_no_vector_activation", False)
    args.gvp_distance_noise = getattr(args, "gvp_distance_noise", 0.0)
    args.gvp_dropout = getattr(args, "gvp_dropout", 0.1)
    args.gvp_edge_hidden_dim_scalar = getattr(args, "gvp_edge_hidden_dim_scalar", 32)
    args.gvp_edge_hidden_dim_vector = getattr(args, "gvp_edge_hidden_dim_vector", 1)
    args.gvp_edge_input_dim_scalar = getattr(args, "gvp_edge_input_dim_scalar", 34)
    args.gvp_edge_input_dim_vector = getattr(args, "gvp_edge_input_dim_vector", 1)
    args.gvp_eps = getattr(args, "gvp_eps", 0.0001)
    args.gvp_ignore_edges_without_coords = getattr(args, "gvp_ignore_edges_without_coords", True)
    args.gvp_layernorm = getattr(args, "gvp_layernorm", True)
    args.gvp_n_edge_gvps = getattr(args, "gvp_n_edge_gvps", 0)
    args.gvp_n_edge_gvps_first_layer = getattr(args, "gvp_n_edge_gvps_first_layer", 0)
    args.gvp_n_message_gvps = getattr(args, "gvp_n_message_gvps", 3)
    args.gvp_no_edge_orientation = getattr(args, "gvp_no_edge_orientation", False)
    args.gvp_node_hidden_dim_scalar = getattr(args, "gvp_node_hidden_dim_scalar", 128)
    args.gvp_node_hidden_dim_vector = getattr(args, "gvp_node_hidden_dim_vector", 64)
    args.gvp_node_input_dim_scalar = getattr(args, "gvp_node_input_dim_scalar", 7)
    args.gvp_node_input_dim_vector = getattr(args, "gvp_node_input_dim_vector", 3)
    args.gvp_num_encoder_layers = getattr(args, "gvp_num_encoder_layers", 6)
    args.gvp_top_k_neighbors = getattr(args, "gvp_top_k_neighbors", 30)
    args.gvp_vector_gate = getattr(args, "gvp_vector_gate", True)


#蛋白质链的映射方式:直接按照字母表的顺序进行映射
#在对于pdb文件进行chain的编码时，需要额外加二，因为0不被考虑，而1表示配体，所以A链是从2开始的
chain_id_to_int = {chr(ord('A') + i): i for i in range(26)}
chain_id_to_int.update({chr(ord('a') + i): 26 + i for i in range(26)})#考虑小写字母
chain_id_to_int.update({str(i): 52 + i for i in range(20)})#数字编号也加上
def get_chain_id_value(chain_id):
    return chain_id_to_int.get(chain_id, -1)


def encode_protein_chain_mask(chain_mask, embedding_dict, one_hot=False):
    """
    将蛋白质链掩码数组转换为嵌入表示，并可选择是否进行 one-hot 编码。

    Args:
        chain_mask (np.ndarray): 包含蛋白质链掩码字母标记的一维 NumPy 数组。
        embedding_dict (dict): 用于转换的字典
        one_hot (bool, optional): 是否进行 one-hot 编码

    Returns:
        torch.Tensor: 处理后的整型张量。
    """
    # 将字母标记转换为嵌入值
    embedded_mask = np.array([embedding_dict.get(char, 0) for char in chain_mask])

    if one_hot:
        # 进行 one-hot 编码
        num_classes = len(embedding_dict)
        one_hot_mask = F.one_hot(torch.from_numpy(embedded_mask), num_classes=num_classes)
        return one_hot_mask
    else:
        # 返回整型张量
        return torch.from_numpy(embedded_mask).long()


def get_args():
    parser = argparse.ArgumentParser(description='abacusT')
    parser.add_argument('--checkpoint', type=str, help='pretrained ckpt file')
    parser.add_argument('--save_dir', type=str, help='directory for saving the results')
    parser.add_argument("--npy_dir", type=str, help='preprocessed numpy file for target protein')
    parser.add_argument("--root_dir", type=str, help='file that store pdbtm data')
    parser.add_argument("--iter_num", type=int, default=5, help='number of iteration during design')
    parser.add_argument("--suffix", type=str, default='', help='suffix of the directory for saving results')
    parser.add_argument("--temperature", type=float, default=1.0, help='sampling temperature')
    parser.add_argument("--packing_only", action='store_true', default=False, help='only repacking sidechain without designing aatype')
    parser.add_argument("--consider_lig", action='store_true', default=False, help='if consider ligand during design or packing')
    parser.add_argument("--max_lig_num", default=10, type=int, help='maximum number of ligand to be considered')
    parser.add_argument("--batchsize", default=10, type=int, help='number of designs for each target')
    parser.add_argument("--bg_dist_file", type=str, default=False, help='backgroud distribution file')
    parser.add_argument("--bg_dist_key", type=str, default=None, help='key of backgroud distribution file')
    parser.add_argument("--bg_weight", default=0.5, type=float, help='weight of backgroud distribution')
    parser.add_argument("--mem_bg_weight", default=0.0, type=float, help="weight for membrane-depth-based background reweighting (0=disabled)")
    parser.add_argument("--start_from_init", action='store_true', default=False, help='iterate from the aa sequence in PDB')
    parser.add_argument("--fixed_positions", type=str, default=None, help="configuration of fixing amino acid type during design e.g. (A: 2 3 4 5 )")
    parser.add_argument("--designed_positions", type=str, default=None)
    parser.add_argument("--save_traj", action='store_true', default=False, help='save trajectory of design or not')
    parser.add_argument("--write_allatom_model", action='store_true', default=False, help='save allatom PDB file or not')
    parser.add_argument("--mask_mode", type=str, default='aatype_nll', help='mask mode during design (violation or aatype_nll')
    parser.add_argument("--esm_pretrained", type=str, default="", help="pretrained ESM model path")
    parser.add_argument("--PLM_selfcond", default=0, type=int, help='if refine sequence with LM or not')
    parser.add_argument("--diff_T", default=40, type=int)
    parser.add_argument("--PLM_param_dir", type=str, default="/home/liuyf/.cache/torch/hub/checkpoints",  help='selfcond_PLM_param_dir')
    parser.add_argument("--selfcondPLM", default="esm2_t33_650M_UR50D.pt", type=str)
    parser.add_argument("--esm_refinement", action='store_true', default=False, help='no selfcondition but PLM refinement')
    parser.add_argument("--refine_model", type=str, default="esm2_t33_650M_UR50D.pt",  help='LM for refining sequence')

    parser.add_argument("--max_crop", type=int, default=512)
    parser.add_argument("--mask_radius", type=float, default=20.0)
    parser.add_argument("--esm_embedder", action='store_true', default=False)
    parser.add_argument("--max_iter_num", default=4, type=int)
    parser.add_argument("--pretrained_mpnn_ckpt", action='store_true', default=False)
    parser.add_argument("--pretrained_mpnn_ckpt_f", type=str, default='')
    parser.add_argument("--encode_mpnn", action='store_true', default=False)

    parser.add_argument('--nar', action='store_true', default=False)
    parser.add_argument("--lig_neighbor_seq_mask", action='store_true', default=False)
    parser.add_argument("--ligmpnn_init", action='store_true', default=False)
    parser.add_argument("--gvp_arch", type=str,default='vt_medium_with_invariant_gvp')
    parser.add_argument("--pre_prot_mode", default='proteinMPNN', type=str) # pifold, proteinMPNN
    parser.add_argument("--embed_unimol_reprs", action='store_true', default=False)
    parser.add_argument("--merge_mpnn_enc_layer_num", default=3, type=int)
    parser.add_argument("--merge_mpnn_dec_layer_num", default=7, type=int)
    parser.add_argument("--augment_eps", default=0.02, type=float)

    parser.add_argument('--device', type=str, default='cuda',help='cpu / cuda / cuda:0 / cuda:1 …')
    parser.add_argument("--tm_raw", type=str,default='raw')
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.0,
                        help='CFG guidance scale for inference (1.0 = no guidance, >1.0 = stronger membrane conditioning)')
    parser.add_argument("--mem_config", type=str, default='',
                        help='Path to unified membrane config YAML (e.g. scripts/configs/mem_config.yaml)')


    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f'checkpoint loaded: {checkpoint_path}')
    # return ckpt['last_optimizer_state']['state'][0]['step']


def crop_receptor(ca_coord, lig_pos, max_len):
    ca_coord = torch.FloatTensor(ca_coord)
    lig_len = lig_pos.shape[0]
    crop_len = max_len - lig_len
    rec_len = ca_coord.shape[0]
    if rec_len <= crop_len:
        return np.arange(rec_len)

    dist = torch.sqrt(torch.sum((ca_coord[:, None] - lig_pos[None])**2, -1)).min(-1)[0] # P, L, 3
    sort_index = torch.argsort(dist)[:crop_len]
    crop_index = torch.sort(sort_index)[0]
    return crop_index.tolist()


def seq_given_mask(ca_coord, lig_pos, ca_radius=9):
    ca_coord = torch.FloatTensor(ca_coord)

    dist = torch.sqrt(torch.sum((ca_coord[:, None] - lig_pos[None])**2, -1)).min(-1)[0] # P, L, 3
    masked_seq_res = dist < ca_radius
    return masked_seq_res


def get_pdbdata(prot_lig_data_f, consider_lig=True, max_lig_num=3):
    """从npy文件中把文件提取出来，转化成feature的张量字典返回

    Args:
        prot_lig_data_f (_type_): _description_
        consider_lig (bool, optional): _description_. Defaults to True.
        max_lig_num (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    prot_lig_data_dict = np.load(prot_lig_data_f, allow_pickle=True).item()
    prot_data_dict = prot_lig_data_dict['prot']
    lig_data_dict = prot_lig_data_dict['lig']
    if len(lig_data_dict) == 0:
        consider_lig = False
    # chain = list(prot_data_dict['pdbres_idx'])[0]
    # chain = prot_data_dict['pdb_chain_mask'][0]
    pdbcode = os.path.basename(prot_lig_data_f).split('.npy')[0]

    atom_bb = prot_data_dict['atom_bb']
    atom_37 = prot_data_dict['atom_37']
    pdbresidx = prot_data_dict['pdbres_idx']
    sequence = prot_data_dict['sequence']
    prot_len = len(sequence)

    atom_bb = torch.from_numpy(atom_bb).float() # M, L, N, 3
    atom_37 = torch.from_numpy(atom_37).float() # M, L, N, 3
    atom_37_mask = (~torch.all(atom_37 == 0, -1)).float()
    pdbresidx = torch.tensor(pdbresidx).long()
    node_mask = torch.ones(( prot_len,)).float()
    af2_tokens = torch.tensor([raw_restype_order[aatype_str] for aatype_str in sequence]).long()
    tokens = af2_tokens + 4
    frame_results = target_features.get_rigid_groups(af2_tokens, atom_37, atom_37_mask)

    pdb_chain_mask = encode_protein_chain_mask(prot_data_dict['pdb_chain_mask'], embedding_dict= chain_id_to_int,one_hot=False)

    features = {}
    features['coords'] = atom_bb[:, [0, 1, 2, 4]]
    features['coords37'] = atom_37
    features['tokens'] = tokens
    features['node_mask'] = node_mask
    features['residx'] = pdbresidx
    features['chainidx'] = pdb_chain_mask
    features['pdbname'] = pdbcode
    features.update(frame_results)

    lig_prot_features = get_lig_ctx(prot_lig_data_dict['lig'], features, consider_lig=consider_lig, max_lig_num=max_lig_num)

    if consider_lig:
        lig_coords = lig_prot_features['lig_coords']
        lig_atoms_num = lig_coords.shape[0]
        all_atom_dist = torch.sqrt(torch.sum((torch.FloatTensor(atom_37[:, :, None]) - lig_coords[None, None])**2, -1)) # P, 37, X
        all_atom_dist_mask = ((1 - atom_37_mask) * 1e6)[..., None]
        all_atom_dist = all_atom_dist + all_atom_dist_mask
        lig_neighbor_seq_mask_pocket = torch.any(torch.any(all_atom_dist < 5.0, dim=-1), dim=-1)
        lig_prot_features['lig_neighbor_seq_mask_pocket'] = torch.cat( [ lig_neighbor_seq_mask_pocket, torch.ones((lig_atoms_num, )).bool() ])


    return lig_prot_features, prot_lig_data_dict['lig'], prot_lig_data_dict['prot']


def get_lig_ctx(lig_dict, features, consider_lig=True, max_lig_num=3, UNIMOL_REPRS_DIM=512):
    lig_prot_features = {}
    prot_coords = features['coords']
    prot_coords37 = features['coords37']
    prot_tokens = features['tokens']
    prot_mask = features['node_mask']
    prot_residx = features['residx']
    prot_chainidx = features['chainidx']
    prot_chi_angles = features['chi_angles']
    prot_alt_chi_angles = features['alt_chi_angles']
    prot_chi_mask = features['chi_mask']
    prot_backbone_affine_tensor = features['backbone_affine_tensor']
    prot_backbone_angles_sin_cos = features['backbone_angles_sin_cos']
    prot_len = prot_coords.shape[0]

    if consider_lig:
        lig_names = list(lig_dict.keys())
        avail_lig_num = len(lig_dict)

        if (avail_lig_num > 0):
            if (avail_lig_num < max_lig_num):
                max_lig_num = avail_lig_num

            cur_sel_lig_names = np.random.choice(lig_names, max_lig_num, replace=False)
            lig_atom_num = 0
            lig_coords_list = []
            lig_single_list = []
            lig_single_unimol_list = []
            lig_edge_index_list = []
            lig_edge_attr_list = []

            lig_plip_anno_itype_list = []

            for lig_name in cur_sel_lig_names:
                raw_data = lig_dict[lig_name]['data']
                unimol_reprs = lig_dict[lig_name]['unimol_reprs']['atomic_reprs'][0]
                cur_lig_atoms_num = unimol_reprs.shape[0]
                lig_single = make_ligand_node_feature(raw_data['ligand']['x'])
                lig_unimol_single = torch.from_numpy(unimol_reprs)

                lig_edge = raw_data['ligand', 'lig_bond', 'ligand']
                lig_edge_index = lig_edge.edge_index + lig_atom_num
                lig_edge_attr = lig_edge.edge_attr

                lig_single_list.append(lig_single)
                lig_single_unimol_list.append(lig_unimol_single)
                lig_edge_index_list.append(lig_edge_index)
                lig_edge_attr_list.append(lig_edge_attr)
                lig_coords_list.append(raw_data['ligand']['pos'])

                cur_lig_anno_itype = torch.ones(cur_lig_atoms_num)
                lig_plip_anno_itype_list.append(cur_lig_anno_itype)

                lig_atom_num += cur_lig_atoms_num

            plip_anno_itype_list = torch.cat([torch.zeros(prot_len), torch.cat(lig_plip_anno_itype_list)], 0)
            lig_single_list = torch.cat(lig_single_list)
            lig_single_unimol_list = torch.cat(lig_single_unimol_list)
            lig_edge_index_list = torch.cat(lig_edge_index_list, -1)
            lig_edge_attr_list = torch.cat(lig_edge_attr_list)
            lig_coords_list = torch.cat(lig_coords_list)

            lig_atom_positions = torch.zeros((lig_atom_num, 4, 3))
            lig_atom_positions[:, 1] = lig_coords_list
            all_atom_positions = torch.cat([prot_coords, lig_atom_positions], dim=0)
            segment = torch.LongTensor([0] * prot_len + [1] * lig_atom_num)
            frame_len = prot_len + lig_atom_num
            ## single
            rec_single = torch.ones((prot_len,1))
            rec_single_dim = rec_single.shape[-1]
            lig_single_dim = lig_single_list.shape[-1]
            single = merged_lig_single = torch.zeros((frame_len, rec_single_dim + lig_single_dim), dtype=torch.float)
            single[:prot_len, :rec_single_dim] = rec_single
            single[prot_len:, rec_single_dim:] = lig_single_list
            merged_lig_single[prot_len:, rec_single_dim:] = lig_single_list
            single = torch.cat([single, F.one_hot(segment, num_classes=2)], dim=-1)
            ## unimol single
            merged_lig_single_unimol = torch.zeros((frame_len, UNIMOL_REPRS_DIM), dtype=torch.float)
            merged_lig_single_unimol[prot_len:] = lig_single_unimol_list
            ## prot edge
            rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
            st, ed = rec_edge_index[0], rec_edge_index[1]
            is_covalent = (prot_residx[ed] - prot_residx[st]) == 1
            rec_edge_index = rec_edge_index[:, is_covalent]
            rev_edge_index = rec_edge_index.clone()
            rev_edge_index[0] = rec_edge_index[1]
            rev_edge_index[1] = rec_edge_index[0]
            rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
            num_rec_edge = rec_edge_index.shape[1]
            ## lig edge
            edge_index = torch.cat([rec_edge_index, lig_edge_index_list + prot_len], dim=1)
            merged_lig_edge_index = torch.cat([torch.ones_like(rec_edge_index) * -2, lig_edge_index_list + prot_len], dim=1)
            edge_attr = merged_lig_edge_attr = torch.zeros((edge_index.shape[1], lig_edge_attr.shape[1] +1))
            edge_attr[:num_rec_edge, 0] = 1
            edge_attr[num_rec_edge:, 1:] = lig_edge_attr_list
            merged_lig_edge_attr[num_rec_edge:, 1:] = lig_edge_attr_list

            logger.info("residx_changed")

            residx = torch.cat([prot_residx, torch.arange(lig_atom_num ) + prot_residx.max() + 100], dim=0)
            residx_=(residx - torch.min(residx) + 1).tolist()
            residx=torch.tensor(residx_)
            # residx_ = residx
            # residx=residx_.clone().detach()
            lig_mask = torch.ones((frame_len,))
            lig_mask[:prot_len] = 0
            tokens = torch.cat([ prot_tokens, torch.LongTensor([24]*lig_atom_num)], dim=0)
            node_mask = torch.ones((frame_len,))
            prot_node_mask = torch.ones((prot_len,))
            chainidx = torch.zeros((frame_len,)).long()
            chainidx[:prot_len] = prot_chainidx + 1
            chainidx = chainidx + 1

            pad_angles = torch.stack([torch.ones(lig_atom_num, 4), torch.zeros(lig_atom_num, 4)], -1)
            chi_angles = torch.cat([prot_chi_angles, pad_angles], 0)
            alt_chi_angles = torch.cat([prot_alt_chi_angles, pad_angles], 0)
            chi_mask = torch.cat([prot_chi_mask, torch.zeros(lig_atom_num, 4)], 0)
            backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor, torch.ones(lig_atom_num, 4, 12)], 0)
            coords37 = torch.cat([prot_coords37, torch.zeros(lig_atom_num, 37, 3)])
            fake_backbone_angles_sin_cos = torch.stack([torch.ones(lig_atom_num, 3), torch.zeros(lig_atom_num, 3)], -1)
            backbone_angles_sin_cos = torch.cat([prot_backbone_angles_sin_cos, fake_backbone_angles_sin_cos], 0)

        else:
            plip_anno_itype_list = torch.zeros(prot_len)
            all_atom_positions = prot_coords
            residx = prot_residx - torch.min(prot_residx) + 1
            # residx = prot_residx
            node_mask = prot_mask
            lig_mask = torch.zeros((prot_len,))
            merged_lig_single = torch.zeros((prot_len, 200), dtype=torch.float)
            merged_lig_single_unimol = torch.zeros((prot_len, UNIMOL_REPRS_DIM), dtype=torch.float)

            rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
            st, ed = rec_edge_index[0], rec_edge_index[1]
            is_covalent = (residx[ed] - residx[st]) == 1
            rec_edge_index = rec_edge_index[:, is_covalent]
            rev_edge_index = rec_edge_index.clone()
            rev_edge_index[0] = rec_edge_index[1]
            rev_edge_index[1] = rec_edge_index[0]
            rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
            num_rec_edge = rec_edge_index.shape[1]

            edge_index = rec_edge_index
            merged_lig_edge_index = (torch.ones_like(rec_edge_index) * -2) #.transpose(0, 1)
            merged_lig_edge_attr = torch.zeros((edge_index.shape[1], 5))

            tokens = prot_tokens
            chainidx = torch.zeros((prot_len,)).long()
            chainidx[:prot_len] = prot_chainidx + 1
            chainidx = chainidx + 1
            cur_sel_lig_names = []

            chi_angles = torch.cat([prot_chi_angles], 0)
            alt_chi_angles = torch.cat([prot_alt_chi_angles], 0)
            chi_mask = torch.cat([prot_chi_mask], 0)
            backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor], 0)
            coords37 = prot_coords37
            backbone_angles_sin_cos = prot_backbone_angles_sin_cos

    else:
        plip_anno_itype_list = torch.zeros(prot_len)
        all_atom_positions = prot_coords
        # residx = prot_residx
        residx = prot_residx - torch.min(prot_residx) + 1
        node_mask = prot_mask
        lig_mask = torch.zeros((prot_len,))
        merged_lig_single = torch.zeros((prot_len, 200), dtype=torch.float)
        merged_lig_single_unimol = torch.zeros((prot_len, UNIMOL_REPRS_DIM), dtype=torch.float)

        rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
        st, ed = rec_edge_index[0], rec_edge_index[1]
        is_covalent = (residx[ed] - residx[st]) == 1
        rec_edge_index = rec_edge_index[:, is_covalent]
        rev_edge_index = rec_edge_index.clone()
        rev_edge_index[0] = rec_edge_index[1]
        rev_edge_index[1] = rec_edge_index[0]
        rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
        num_rec_edge = rec_edge_index.shape[1]

        edge_index = rec_edge_index
        merged_lig_edge_index = (torch.ones_like(rec_edge_index) * -2) #.transpose(0, 1)
        merged_lig_edge_attr = torch.zeros((edge_index.shape[1], 5))

        tokens = prot_tokens
        chainidx = torch.zeros((prot_len,)).long()
        chainidx[:prot_len] = prot_chainidx + 1
        chainidx = chainidx + 1
        cur_sel_lig_names = []

        chi_angles = torch.cat([prot_chi_angles], 0)
        alt_chi_angles = torch.cat([prot_alt_chi_angles], 0)
        chi_mask = torch.cat([prot_chi_mask], 0)
        backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor], 0)
        coords37 = prot_coords37
        backbone_angles_sin_cos = prot_backbone_angles_sin_cos

    lig_prot_features['tokens'] =  tokens
    plip_anno_itype_list = torch.zeros(tokens.shape[0])
    lig_prot_features['coords'] =  all_atom_positions
    lig_prot_features['node_mask'] =  node_mask
    lig_prot_features['lig_mask'] =  lig_mask
    lig_prot_features['residx'] =  residx
    lig_prot_features['lig_node_attr'] =  merged_lig_single
    lig_prot_features['unimol_lig_node_attr'] =  merged_lig_single_unimol
    lig_prot_features['lig_edge_index'] =  merged_lig_edge_index.transpose(0, 1)
    lig_prot_features['lig_edge_attr'] =  merged_lig_edge_attr
    lig_prot_features['chainidx'] =  chainidx
    lig_prot_features['pdbname'] =  features['pdbname']
    lig_prot_features['lignames'] =  cur_sel_lig_names
    lig_prot_features['plip_anno_itype_list'] =  plip_anno_itype_list

    lig_prot_features['chi_angles'] = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
    lig_prot_features['alt_chi_angles'] = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
    lig_prot_features['chi_mask'] = chi_mask
    lig_prot_features['backbone_affine_tensor'] = backbone_affine_tensor
    lig_prot_features['backbone_angles_sin_cos'] = backbone_angles_sin_cos
    lig_prot_features['coords37'] = coords37

    try:
        lig_prot_features['lig_coords'] = lig_coords_list
    except:
        pass

    return lig_prot_features



def expand_batch(batch: dict, expand_size: int):
    new_batch = {}
    for k in batch.keys():
        if isinstance(batch[k], list):
            new_batch[k] = batch[k] * expand_size
        else:
            shape_len = len(batch[k].shape)-1
            repeat_shape = [expand_size] + [1] * shape_len
            new_batch[k] = batch[k].repeat(*repeat_shape)

    return new_batch


def collater(samples, device=torch.device('cuda')):
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        return None
    batch = {}
    max_len = max([sample['coords'].shape[0] for sample in samples])
    for k in samples[0].keys():
        if k in ['tokens']:
            batch[k] = pad_and_stack([s[k] for s in samples], dim=0, value=1).to(device, non_blocking=True)
        elif k in ['lig_neighbor_seq_mask']:
            batch[k] = pad_and_stack([s[k] for s in samples], dim=0, value=True).to(device, non_blocking=True)
        elif k in ['lig_edge_index']:
            batch[k] = pad_and_stack([s[k] for s in samples], dim=0, value=-2).to(device, non_blocking=True)
        elif k in ['pdbname', 'lignames']:
            batch[k] = [s[k] for s in samples]
        elif k in ['rec_len']:
            batch[k] = [s[k] for s in samples]
        else:
            try:
                batch[k] = pad_and_stack([s[k] for s in samples], dim=0, value=0).to(device, non_blocking=True)
            except:
                import pdb;pdb.set_trace()
    return batch


def print_stat(test_all_identity, test_pocket_identity, prefix=''):
    stat_mean_all_ident = torch.cat(test_all_identity, -1).max(0)[0].median(-1)[0]
    stat_mean_pocket_ident = torch.cat(test_pocket_identity, -1).max(0)[0].median(-1)[0]
    # stat_mean_all_nll = torch.cat(test_nll, -1).mean(-1)

    logger.info(f'{prefix} avg all identity: {stat_mean_all_ident}')
    logger.info(f'{prefix} avg pocket identity: {stat_mean_pocket_ident}')
    # logger.info(f'avg all nll: {stat_mean_all_nll}')


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



def esm_refine(pred_seqs, esm_batcher, esm_model, esm_alphabet, device):
    """Use ESM-1b to refine model predicted"""

    _, _, input_ids = esm_batcher(
        [('_', seq) for seq in pred_seqs]
    )

    # input_ids = pred_ids
    results = esm_model(input_ids.to(device), repr_layers=[33], return_contacts=False)
    logits = results['logits'].detach().cpu()
    refined_ids = logits.argmax(-1)[..., 1:-1]

    refined_seq = [''.join([esm_alphabet.get_tok(token.item()) for token in tokens]) for tokens in refined_ids]

    return refined_seq



# def TMalign_structure(align_mobile_mc, align_target_mc, mobile_aa, target_aa, mobile_tmp_f, target_tmp_f, mobile_mc=None):
#     """把两条蛋白质主链（mobile 与 target）做结构叠合（structure alignment），并把 mobile 链通过旋转-平移矩阵变换到 target 坐标系**的实用函数。它内部调用了著名的 TM-align 算法来计算叠合矩阵，

#     Returns:
#         _type_: _description_
#     """
#     write_multichain_from_atoms([align_mobile_mc.reshape(-1, 3)], mobile_tmp_f, aatype=[mobile_aa])
#     write_multichain_from_atoms([align_target_mc.reshape(-1, 3)], target_tmp_f, aatype=[target_aa])
#     mobile_filename = tempfile.mktemp('.mtx', 'tm')
#     tmscore, rmsd, alignment, aligned_len = tmaligner.run(mobile_tmp_f, target_tmp_f, 'A', 'A', matrix_file=mobile_filename)

#     rotrans_mtx = parse_matrixfile(mobile_filename)
#     trans_vec = rotrans_mtx[:, 0]
#     rot_mtx = rotrans_mtx[:, 1:].transpose(1, 0)
#     if (mobile_mc is None):
#         mobile_mc = align_mobile_mc
#     rotransed_mobile = ((mobile_mc.reshape(-1, 3) @ rot_mtx) + trans_vec).reshape(-1, 4, 3)

#     return rotransed_mobile, tmscore, rmsd, alignment, aligned_len
    ## debug
    # rotransed_mobile_f = tempfile.mktemp('.pdb', 'rotransed_mobile')
    # # rotransed_mobile_f = 'rotransed_mobile_T.pdb'
    # write_multichain_from_atoms([rotransed_mobile.reshape(-1, 3)], rotransed_mobile_f)


# def compute_rmsf(ref_traj, mobile_gen_data_trajs, init_residx_range='all'):
#     # ref_traj.shape L, 4, 3
#     # mobile_gen_data_trajs.shape M, L, 4, 3
#     res_num = ref_traj.shape[0]
#     assert ref_traj.shape[0] == mobile_gen_data_trajs.shape[1]

#     residx_aligned_coords = []
#     if (init_residx_range == 'all'):
#         residx_range = np.arange(res_num)
#     else:
#         residx_range = init_residx_range

#     for gen_traj_crd in mobile_gen_data_trajs:
#         rotransed_mobile_f = tempfile.mktemp('.pdb', 'mobile_tmp')
#         target_tmp_f = tempfile.mktemp('.pdb', 'target_tmp')
#         aligned_traj_mc_coords = TMalign_structure(
#             gen_traj_crd[residx_range], ref_traj[residx_range], 'A' * len(residx_range), 'A' * len(residx_range),
#             rotransed_mobile_f, target_tmp_f, mobile_mc=gen_traj_crd)[0]
#         residx_aligned_coords.append(aligned_traj_mc_coords)

#     residx_aligned_coords = np.stack(residx_aligned_coords)

#     ca_residx_aligned_coords = residx_aligned_coords[..., 1, :]
#     mean_residue_pos = ca_residx_aligned_coords.mean(0)
#     squared_distance = np.sum((ca_residx_aligned_coords - mean_residue_pos[None])**2, axis=-1)**0.5
#     rmsf_values= np.mean(squared_distance**2, 0) ** 0.5

#     return rmsf_values



def get_pdbtm_data(batch_size = 1,align = None, device=torch.device('cuda')):
    """获取文件目录中的json文件，这些json文件表征的是蛋白质的跨膜信息，包括regions，放射矩阵G，以及膜的法向量N
    在输入的时候应该确保输入的regions和原来蛋白质文件的蛋白部分是对齐的（代码里暂时没有检查步骤，只是简单的长度补齐）

    Returns:
        _type_: 返回regions，G，N

    """
    # print( os.listdir(args.root_dir))
    flist = os.listdir(args.root_dir)#pdb的文件和pdbtm的文件放在一个文件夹里
    pdbtm_json_list = [f for f in flist if f.endswith('.json')]#筛选出所有以pdbtm结尾的文件
    # import pdb;pdb.set_trace()
    try:
        file_path = os.path.join(args.root_dir, pdbtm_json_list[0])
        with open(f"{file_path}", "r") as f:#这里先假设是一个一个文件设计
            data = json.load(f)
    except:
        logger.warning(f"file {file_path} not found, generate an empty tensor instead")
        import pdb;pdb.set_trace()
        return None
    # 赋值
    region_str = data["region"]          # str
    G_list = data["G"]               # list[list[float]]
    N_list = data["N"]               # list[float]

    region = list(region_str)           # 转换成list
    if not region:
        raise ValueError("region could not be empty")
    # 统一检查 + 映射
    try:
        region = [region_dict[c] for c in region]
    except KeyError as e:
        raise KeyError(f"字符 {e} 不在 region_dict 中") from None
    try:
        region_tensor = torch.tensor(region, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)

        # 转成张量
        G = torch.tensor(G_list, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # shape (B,3,4)
        N = torch.tensor(N_list, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(device)   # shape (B,3,)
    except Exception as e:
        traceback.print_exc()
        import pdb;pdb.set_trace()

    if align is not None:
        assert isinstance(align, torch.Tensor), f"align 必须是 torch.Tensor，实际类型: {type(align)}"
        L_align   = align.size(1)
        L_region  = region_tensor.size(1)


        assert L_align >= L_region, (f"align 的 seq_len ({L_align}) 必须 ≥ region_tensor 的 seq_len ({L_region})")
        pad_len = L_align - L_region
        region_tensor = torch.nn.functional.pad(region_tensor, (0, pad_len, 0, 0), value=0)
        # import pdb;pdb.set_trace()


    return (region_tensor, G, N)




def main(args):
    base_architecture(args)
    device = torch.device(args.device)
    logger.info(f'Running on device: {device}')

    model = DiffFullAtom(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    esm_refinement = args.esm_refinement
    if esm_refinement:
        from esm import pretrained
        ESM_PRETRAIN = "/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

        esm_model, esm_alphabet = pretrained.load_model_and_alphabet(ESM_PRETRAIN)
        esm_batcher = esm_alphabet.get_batch_converter()
        esm_model.to(device)
        logger.info(f'esm model loaded: {ESM_PRETRAIN}')
    else:
        esm_model = None
        esm_alphabet = None
        esm_batcher = None

    designer = model.model.abacust

    iter_num=args.iter_num #5
    temperature = args.temperature #1.0
    refined_test_all_identity = []
    refined_test_pocket_identity = []
    test_all_identity = []
    test_all_violation_num = []
    test_pocket_identity = []
    max_lig_num = args.max_lig_num
    batchsize = args.batchsize
    npy_list = os.listdir(args.npy_dir)

    if args.bg_dist_file is not None and args.bg_dist_key is not None:#背景文件（什么玩意？）
        bg_dict = np.load(args.bg_dist_file, allow_pickle=True).item()
        bg_dist = torch.softmax(bg_dict[args.bg_dist_key], -1)
    else:
        bg_dist = None

    with torch.no_grad():
        for l_idx, f_ in enumerate(npy_list):
            logger.info(f'{l_idx}/{len(npy_list)}: {f_}')
            if not f_.endswith('.npy'):
                continue
            else:
                try:
                    data_name = os.path.basename(f_).split('.npy')[0]

                    npy_f = f'{args.npy_dir}/{data_name}.npy'#拼接路径
                    features, lig_raw_dict, prot_raw_dict = get_pdbdata(npy_f, consider_lig=args.consider_lig, max_lig_num=max_lig_num)

                    #加载相应的features字典等，后两项分别是npy文件直接解析出来的东西，即lig还有prot部分

                    batch = collater([features], device = device)
                    batch = expand_batch(batch, batchsize)#组合成一个批次


                    merged_coords = batch['coords']
                    merged_s = batch['tokens']
                    merged_mask = batch['node_mask']

                    B, L = merged_coords.shape[:2]
                    merged_chain_mask = batch['chainidx']
                    merged_residue_idx = batch['residx']
                    print(batch["residx"])

                    if bg_dist is not None:
                        bg_dist_prob = torch.zeros((1, L, 35)).to(device)
                        for residx in range(bg_dist.shape[0]):
                            for aa in raw_restypes:
                                bg_dist_prob[0, residx, restype_order[aa]] = bg_dist[residx, esm_dict[aa]]
                        bg_dist_prob = bg_dist_prob/(torch.sum(bg_dist_prob, -1, keepdim=True) + 1e-8)
                    else:
                        bg_dist_prob = None

                    merged_chain_encoding_all = batch['chainidx']
                    # randn = torch.randn_like(merged_chain_mask)

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
                    backbone_affine_tensor = batch['backbone_affine_tensor']
                    backbone_angles_sin_cos = batch['backbone_angles_sin_cos']
                    plip_anno_itype_list = batch['plip_anno_itype_list']

                    if args.tm_raw.startswith("zero") or args.tm_raw.startswith("raw"):
                        pdbtm_regions = torch.zeros_like(merged_s)
                        G = torch.zeros(batchsize, 3, 4, dtype=torch.float32, device=device)
                        N = torch.zeros(batchsize, 3, dtype=torch.float32, device=device)
                        if args.mem_bg_weight > 0:
                            try:
                                _, G, N = get_pdbtm_data(batch_size=batchsize, align=merged_s, device=device)
                            except Exception:
                                pass
                    else:
                        pdbtm_regions, G, N = get_pdbtm_data(batch_size=batchsize, align=merged_s, device=device)

                    try:
                        seq_mask_pocket = batch['lig_neighbor_seq_mask_pocket']
                    except:
                        seq_mask_pocket = torch.zeros_like(merged_s)

                    initial_S = None
                    return_pdb_ctx_iter = np.arange(iter_num)
                    design_ctx = designer.nar_sample(
                        merged_coords, merged_s, merged_mask, merged_chain_mask, merged_residue_idx, merged_chain_encoding_all,
                        lig_node_attr,
                        lig_edge_attr,
                        lig_edge_index,
                        pdbtm_regions = pdbtm_regions,
                        G = G,
                        N = N,
                        lig_mask=merged_lig_mask,
                        backbone_affine_tensor=backbone_affine_tensor, backbone_angles_sin_cos=backbone_angles_sin_cos,
                        iter_num=iter_num, temperature=temperature, seq_mask_pocket=seq_mask_pocket,
                        unimol_reprs=unimol_reprs,
                        packing_only=args.packing_only,
                        esm_batcher=esm_batcher, esm_model=esm_model, esm_alphabet=esm_alphabet, #这几个esm参数由esmrefinement决定
                        mask_mode=args.mask_mode,
                        return_pdb_ctx_iter=return_pdb_ctx_iter,
                        start_from_init=args.start_from_init,
                        initial_S=initial_S,
                        bg_dist=bg_dist_prob,
                        bg_weight=args.bg_weight,
                        plip_anno_itype_list=plip_anno_itype_list,
                        cfg_guidance_scale=args.cfg_guidance_scale,
                        mem_bg_weight=args.mem_bg_weight,
                        )



                    # import pdb;pdb.set_trace()
                    subdir = f'{args.save_dir}/T_{temperature}_R_{iter_num}_{args.suffix}/{data_name}'
                    os.makedirs(subdir, exist_ok=True)

                    iter_seq_list = design_ctx['aatype']
                    chain_mask = batch['chainidx']

                    # import pdb;pdb.set_trace()

                    chain_mask  = chain_mask[0]
                    chain_idx = [idx - 1 for idx in range(1, len(chain_mask)) if chain_mask[idx] != chain_mask[idx - 1]]
                    chain_idx.append(len(chain_mask)-1)
                    chain_list = [((0 if i == 0 else chain_idx[i-1]+1),n) for i,n in enumerate(chain_idx)]



                    nat_seq_list = [raw_res_id_to_aatype[(aatype-4).item()] for aatype in merged_s[0][[design_ctx['protein_mask'][0].bool()]] ]
                    nat_seq_raw = ''.join(nat_seq_list)
                    nat_seq_region_list = [nat_seq_raw[start:end+1] for start,end in chain_list]
                    nat_seq = ':'.join(nat_seq_region_list)
                    log_dict = {}

                    log_dict[data_name] = {'comment': f'>{data_name}', 'seq': nat_seq}
                    overall_cur_ident_list = []
                    for iter_idx in return_pdb_ctx_iter:
                        for b_idx in range(iter_seq_list.shape[1]):
                            design_seq_list = [raw_res_id_to_aatype[(aatype-4).item()] for aatype in iter_seq_list[iter_idx, b_idx][design_ctx['protein_mask'][b_idx].bool()]]
                            design_seq_raw = ''.join(design_seq_list)

                            design_seq_region_list = [design_seq_raw[start:end+1] for start,end in chain_list]
                            design_seq = ':'.join(design_seq_region_list)

                            overall_cur_ident = design_ctx['all_identity'][iter_idx, b_idx].item()
                            overall_cur_ident_list.append(overall_cur_ident)
                            pocket_cur_ident = design_ctx['pocket_identity'][iter_idx, b_idx].item()

                            log_dict[f'{data_name}_design_{b_idx}_{iter_idx}'] = {
                                'comment': f'>{data_name}_design_{b_idx}_{iter_idx}; overall_identity={round(overall_cur_ident,3)}; pocket_identity={round(pocket_cur_ident,3)}',
                                'seq': design_seq}
                    torsion_continuous = design_ctx['torsion_continuous']
                    overall_cur_ident_list = overall_cur_ident_list[-batchsize:]
                    # print(overall_cur_ident_list)
                    mean_val = sum(overall_cur_ident_list) / batchsize
                    var_val  = math.sqrt(sum((x - mean_val) ** 2 for x in overall_cur_ident_list) / batchsize)   # 总体方差
                    mean_val = round(mean_val,4)
                    var_val = round(var_val,4)

                    gt_pdb_ctx = make_pdb_ctx_from_design(merged_s, torsion_angles, backbone_angles_sin_cos, backbone_affine_tensor)
                    absl_nat_pdb_f = f'{subdir}/{data_name}.pdb'
                    prot = from_pdb_string(gt_pdb_ctx)
                    struct_metrics = get_violation_metrics(prot, clash_overlap_tolerance=1.0)
                    gt_num_residue_violations = struct_metrics['num_residue_violations']
                    log_dict[data_name]['comment'] = log_dict[f'{data_name}']['comment']

                    if args.write_allatom_model:
                        with open(absl_nat_pdb_f, 'w') as writer:
                            writer.write(gt_pdb_ctx)

                        for b_idx in range(iter_seq_list.shape[1]):
                            violation_list = []
                            for iter_idx in return_pdb_ctx_iter:
                                pdb_ctx = make_pdb_ctx_from_design(
                                    iter_seq_list[iter_idx], torsion_continuous[iter_idx].cpu(),
                                    backbone_angles_sin_cos.cpu(), backbone_affine_tensor.cpu(), b_idx)
                                prot = from_pdb_string(pdb_ctx)
                                struct_metrics = get_violation_metrics(prot, clash_overlap_tolerance=1.0)
                                num_residue_violations = (struct_metrics['num_residue_violations'] - gt_num_residue_violations)
                                num_residue_violations = num_residue_violations if num_residue_violations >=0 else 0
                                violation_list.append(num_residue_violations)
                                log_dict[f'{data_name}_design_{b_idx}_{iter_idx}']['comment'] = log_dict[f'{data_name}_design_{b_idx}_{iter_idx}']['comment'] + f'; violation={num_residue_violations}'
                                # 要想恢复生成pdb，还原下面即可
                                if iter_idx == return_pdb_ctx_iter[-1]:
                                    absl_design_pdb_f = f'{subdir}/{data_name}_design_{b_idx}_{iter_idx}.pdb'
                                    with open(absl_design_pdb_f, "w") as writer:
                                        writer.write(pdb_ctx)

                        logger.info(f'all atoms model has been saved in {subdir}')
                        #书写配体文件
                        # for ligname in features['lignames']:
                        #     mol = lig_raw_dict[ligname]['rdkit_ligand']
                        #     Chem.MolToMolFile(mol, f'{subdir}/{ligname}')


                    #保存fasta文件
                    with open(f'{subdir}/{data_name}_design.fa', 'w') as writer:
                        for protein, prot_ctx in log_dict.items():
                            writer.write('\n'.join([prot_ctx['comment'], prot_ctx['seq']]) + '\n')

                    if "mem_position" in design_ctx:
                        write_mem_position(design_ctx, subdir)

                    logger.info(f'{data_name} design finished')
                    logger.info(f"mean_val = {mean_val}")
                    logger.info(f"var_val = {var_val}")


                except RuntimeError as e:
                    import traceback;traceback.print_exception(e)

                    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args()
    main(args)


def write_mem_position(design_ctx, subdir ):
    assert design_ctx.get('key') is not None ,"design_ctx's position attribute is none"
    #保存蛋白质氨基酸相对膜的位置
    mem_position = design_ctx["mem_position"]
    mem_position_np = mem_position.detach().cpu().numpy()
    np.save(os.path.join(subdir, 'position.npy'), mem_position_np)
