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
from pdbtm_parser import Pdbtm_parser
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
sys.path.append("/home/chenty/abacust_mem/src/")

from rdkit import Chem
from fasde.models.diff_full_atom import DiffFullAtom
from fasde.modules.design_utils import  make_pdb_ctx_from_design

from fasde.data.utils.diffusion_utils_atoms import make_complex_feature, make_ligand_node_feature
from fasde.data.utils import features,target_features
# from data_utils import pad_and_stack
from fasde.utils.relax.assess_violation import get_violation_metrics
from fasde.modules.alphafold.common.protein import from_pdb_string
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID
import traceback
from protein_utils.pdbtm_data_parser import Pdbtm_parser

# from pytmalign import TMalign, parse_matrixfile
# tmaligner = TMalign()

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s| %(name)s | %(message)s",
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

import pynvml

def log_gpu():
    pynvml.nvmlInit()

    logical_device_idx = torch.cuda.current_device()
    visible_devices_str = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    visible_devices = [int(x) for x in visible_devices_str.split(',')] if visible_devices_str else list(range(torch.cuda.device_count()))
    physical_device_idx = visible_devices[logical_device_idx]

    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_idx)
    name = pynvml.nvmlDeviceGetName(handle)
    uuid = pynvml.nvmlDeviceGetUUID(handle)

    # 如果是bytes类型，直接用str()转，不调用decode()
    if isinstance(name, bytes):
        name = str(name, 'utf-8', errors='ignore')
    if isinstance(uuid, bytes):
        uuid = str(uuid, 'utf-8', errors='ignore')

    print(f"[log_gpu] torch.cuda.current_device(): cuda:{logical_device_idx}")
    print(f"[log_gpu] CUDA_VISIBLE_DEVICES={visible_devices_str}")
    print(f"[log_gpu] 物理设备索引: {physical_device_idx}")
    print(f"[log_gpu] NVML GPU 名称: {name}")
    print(f"[log_gpu] NVML UUID: {uuid}")

    pynvml.nvmlShutdown()
    return physical_device_idx

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def pad_to_length(data, max_len, dim=0,value=0):
    shape = [d for d in data.shape]
    assert(shape[dim] <= max_len)
    if shape[dim] <= max_len:
        return data

    pad_len = max_len-shape[dim]
    pad_shape = [[0, 0]] * dim + [[0, pad_len]] + [[0, 0]] * (len(shape) - dim -1)
    data_pad = F.pad(data,convert_pad_shape(pad_shape),mode='constant', value=value)
    return data_pad
def pad_and_stack(data_list, dim=0,value=0):
    max_len = max([d.shape[dim] for d in data_list])
    data = torch.stack(
        [pad_to_length(d, max_len, dim, value) for d in data_list], dim =0)
    return data

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
    parser.add_argument("--freeze_encoder_param", action='store_true', default=False)

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

    parser.add_argument('--device', type=str, default='cuda:0',help='cpu / cuda / cuda:0 / cuda:1 …')
    parser.add_argument("--tm_raw", type=str,default='raw')
    parser.add_argument("--mem_config", type=str, default='',
                        help='Path to unified membrane config YAML')
    parser.add_argument("--cfg_guidance_scale", type=float, default=1.0,
                        help='CFG guidance scale (1.0 = no guidance)')
    parser.add_argument("--data_list", type=str, default="")
    parser.add_argument("--order_suffix", type=str, default="1")
    parser.add_argument("--esm_pretrained", type=str, default="")


    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    has_gate_encoder_weights = any(
        "mem_logit_gate_encoder_layers." in key for key in ckpt["model"]
    )
    abacust = getattr(getattr(model, "model", None), "abacust", None)
    if (
        abacust is not None
        and getattr(abacust, "mem_logit_gate_modulation", False)
        and getattr(abacust, "mem_logit_gate_encoder_layers", None) is not None
        and not has_gate_encoder_weights
    ):
        abacust.initialize_mem_logit_gate_encoder_from_encoder()
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


########################################
# [新增] 2026-03-10 cty
# 针对超长蛋白（>1800残基）的几何crop函数。
# 随机选一个残基作为空间中心，取CA坐标最近的k个残基，按原序返回索引。
# 不依赖配体位置，与口袋crop（crop_receptor）逻辑分离。
########################################
def crop_protein_knn(ca_coord, k):
    """随机选一个残基作为中心，取空间上最近的k个残基的索引（保序）"""
    ca_coord = torch.FloatTensor(ca_coord)
    prot_len = ca_coord.shape[0]
    if prot_len <= k:
        return np.arange(prot_len)
    center_idx = torch.randint(0, prot_len, (1,)).item()
    center = ca_coord[center_idx]  # [3]
    dist = torch.sqrt(torch.sum((ca_coord - center[None]) ** 2, -1))  # [P]
    sort_index = torch.argsort(dist)[:k]
    crop_index = torch.sort(sort_index)[0]
    return crop_index.tolist()
########################################


def seq_given_mask(ca_coord, lig_pos, ca_radius=9):
    ca_coord = torch.FloatTensor(ca_coord)

    dist = torch.sqrt(torch.sum((ca_coord[:, None] - lig_pos[None])**2, -1)).min(-1)[0] # P, L, 3
    masked_seq_res = dist < ca_radius
    return masked_seq_res


########################################
# [修改] 2026-03-10 cty
# 新增 max_len 参数（默认1800），超长蛋白在进入 get_lig_ctx 前先 crop，
# 保证配体完整保留（crop 在 prot-only features 上做，lig 由 get_lig_ctx 接回）。
########################################
def get_pdbdata(prot_lig_data_f, consider_lig=True, max_lig_num=3, max_len=1800):
########################################
    """从npy文件中把文件提取出来，转化成feature的张量字典返回

    Args:
        prot_lig_data_f (_type_): _description_
        consider_lig (bool, optional): _description_. Defaults to True.
        max_lig_num (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    try:
        prot_lig_data_dict = np.load(prot_lig_data_f, allow_pickle=True).item()
    except:
        file_path = Path(prot_lig_data_f)
        dir_path = file_path.parent  # 获取目录部分
        file_name = file_path.stem  # 获取不带扩展名的文件名
        prefix = file_name[:4]
        files_with_prefix = [f for f in dir_path.iterdir() if f.name.startswith(prefix) and f.is_file()][0]
        prot_lig_data_dict = np.load(files_with_prefix, allow_pickle=True).item()
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

    ########################################
    # [新增] 2026-03-10 cty
    # 蛋白长度超过 max_len 时，用 crop_protein_knn 做几何 crop（配体不参与）。
    # features 中 shape[0]==prot_len 的所有张量同步裁剪，atom_37/atom_37_mask
    # 作为局部变量单独裁剪（供后续 lig_neighbor_seq_mask_pocket 计算使用）。
    # crop 完成后再调用 get_lig_ctx，配体原子完整接回。
    ########################################
    if max_len is not None and prot_len > max_len:
        ca_coord = features['coords'][:, 1, :]  # [prot_len, 3] CA coordinates
        crop_idx = crop_protein_knn(ca_coord, max_len)
        for k, v in features.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == prot_len:
                features[k] = v[crop_idx]
        atom_37 = atom_37[crop_idx]
        atom_37_mask = atom_37_mask[crop_idx]
        prot_len = len(crop_idx)
        logging.info(f"{pdbcode} length over {max_len}, cropped to {prot_len}")
    ########################################

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



def main(args):
    base_architecture(args)
    device = torch.device(args.device)
    logger.info(f'Running on device: {device}')
    logger.info(f"[env] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    physical_device_idx = log_gpu() if device.type == "cuda" else -1  # CPU/无CUDA 时跳过, 避免 torch.cuda.current_device() 崩
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
    max_lig_num = args.max_lig_num
    batchsize = args.batchsize
    npy_list = args.data_list.split(":")

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

                    origin_batch = collater([features], device = device)
                    coords_shape = origin_batch["coords"].shape[1]

                    def calculate_epochs(batch_size, single_batch_size):
                        # 计算多少轮 epochs 才能使得 single_batchsize * epochs >= batch_size
                        return math.ceil(batch_size / single_batch_size)

                    ########################################
                    # [修改] 2026-03-10 cty
                    # 原来 (1800, None) 条目会直接跳过超长蛋白，现已移除。
                    # 超长蛋白由 get_pdbdata 中的 crop_protein_knn 处理，
                    # crop 后长度 <=1800，走 (1000, 2) 分支，batchsize=2。
                    ########################################
                    # 定义映射表，包含阈值和对应的 single_batchsize
                    thresholds = [
                        (1000, 2),     # 如果大于等于 1000，single_batchsize 设为 2
                        (700, 3),      # 如果大于等于 700，single_batchsize 设为 3
                        (500, 4),      # 如果大于等于 500，single_batchsize 设为 4
                        (200, 5),      # 如果大于等于 200，single_batchsize 设为 5
                        (0, 10)        # 如果小于 200，single_batchsize 设为 10
                    ]
                    ########################################

                    single_batchsize = None
                    for threshold, batch_size in thresholds:
                        if coords_shape >= threshold:
                            single_batchsize = min(batch_size, batchsize)
                            epochs = calculate_epochs(batchsize, single_batchsize)
                            break  # 找到匹配的条件后退出循环


                    # 输出最终的 batchsize 和 epochs
                    logging.info(f"single_batchsize: {single_batchsize}, epochs: {epochs}")
                    subdir = f'{args.save_dir}/T_{temperature}_R_{iter_num}_{args.suffix}/{data_name}'
                    import shutil
                    shutil.rmtree(subdir, ignore_errors=True)   # 不存在也不抛异常
                    os.makedirs(subdir, exist_ok=True)


                    for index in range(epochs):
                        batch = expand_batch(origin_batch, single_batchsize)#组合成一个批次



                        merged_coords = batch['coords']
                        merged_s = batch['tokens']
                        merged_mask = batch['node_mask']


                        B, L = merged_coords.shape[:2]
                        merged_chain_mask = batch['chainidx']
                        merged_residue_idx = batch['residx']
                        # print(batch["residx"])

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

                        unimol_reprs = batch['unimol_lig_node_attr']
                        torsion_angles=batch['chi_angles']
                        alt_chi_angles = batch['alt_chi_angles']
                        torsion_angle_mask=batch['chi_mask']
                        backbone_affine_tensor = batch['backbone_affine_tensor']
                        backbone_angles_sin_cos = batch['backbone_angles_sin_cos']
                        plip_anno_itype_list = batch['plip_anno_itype_list']

                        if args.tm_raw.startswith("zero") or args.tm_raw.startswith("raw"):
                            pdbtm_regions = None
                            G = None
                            N = None
                            if args.mem_bg_weight > 0 or designer.mem_logit_gate_modulation or designer.mem_logit_modulation:
                                try:
                                    pdbtm_npy_path = Path('/home/chenty/public_data/tmpdb_after_202505/abacust_design_result/original_output/npys_tmdet') / f'{data_name}.npy'
                                    pdbtm_data_item = Pdbtm_parser._init_from_filepath(str(pdbtm_npy_path))
                                    G = torch.as_tensor(pdbtm_data_item._tmatrix, device=device, dtype=merged_coords.dtype).unsqueeze(0).repeat(single_batchsize, 1, 1)
                                    N = torch.as_tensor(pdbtm_data_item._normal, device=device, dtype=merged_coords.dtype).unsqueeze(0).repeat(single_batchsize, 1)
                                except Exception:
                                    pass
                        else:
                            pdbtm_npy_path = Path('/home/chenty/public_data/tmpdb_after_202505/abacust_design_result/original_output/npys_tmdet') / f'{data_name}.npy'
                            assert pdbtm_npy_path.exists(), f"{pdbtm_npy_path} dose not exist! "

                            pdbtm_data_item = Pdbtm_parser._init_from_filepath(str(pdbtm_npy_path))
                            G = torch.as_tensor(pdbtm_data_item._tmatrix, device=device, dtype=merged_coords.dtype).unsqueeze(0).repeat(single_batchsize, 1, 1)
                            N = torch.as_tensor(pdbtm_data_item._normal, device=device, dtype=merged_coords.dtype).unsqueeze(0).repeat(single_batchsize, 1)
                            valid_pdbtm_regions = None
                            pdbtm_regions = None
                            # import ipdb; ipdb.set_trace()

                        try:
                            seq_mask_pocket = batch['lig_neighbor_seq_mask_pocket']
                        except:
                            seq_mask_pocket = torch.zeros_like(merged_s)

                        initial_S = None
                        return_pdb_ctx_iter = np.arange(iter_num)

                        mask_mode=args.mask_mode

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

                        if os.environ.get('ABACUST_MEM_DEBUG_DUMP', '0') == '1' and design_ctx.get('mem_logit_debug') is not None:
                            debug_dump_path = f'{subdir}/{data_name}_{index}_mem_logit_debug.pt'
                            debug_payload = {
                                'meta': {
                                    'data_name': data_name,
                                    'checkpoint': args.checkpoint,
                                    'tm_raw': args.tm_raw,
                                    'suffix': args.suffix,
                                    'iter_num': iter_num,
                                    'temperature': temperature,
                                    'batchsize': batchsize,
                                    'single_batchsize': single_batchsize,
                                    'epoch_index': index,
                                },
                                'entries': design_ctx['mem_logit_debug'],
                            }
                            torch.save(debug_payload, debug_dump_path)
                            logging.info(f' mem logit debug has been saved in {debug_dump_path}')

                        iter_seq_list = design_ctx['aatype']
                        chain_mask = batch['chainidx']

                        chain_mask  = chain_mask[0]
                        assert chain_mask.ndimension() == 1, f"Tensor must be one-dimensional!, but got size :{ chain_mask}"
                        chain_set = list(dict.fromkeys(chain_mask.tolist()))

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
                                    'comment': f'>{data_name}_design_{b_idx}_{iter_idx}; overall_identity={round(overall_cur_ident,3)}; pocket_identity={round(pocket_cur_ident,3)};chains={chain_set}',
                                    'seq': design_seq}
                        # torsion_continuous = design_ctx['torsion_continuous']
                        overall_cur_ident_list = overall_cur_ident_list[-single_batchsize:]
                        # print(overall_cur_ident_list)
                        mean_val = sum(overall_cur_ident_list) / single_batchsize
                        var_val  = math.sqrt(sum((x - mean_val) ** 2 for x in overall_cur_ident_list) / single_batchsize)   # 总体方差
                        mean_val = round(mean_val,4)
                        var_val = round(var_val,4)

                        gt_pdb_ctx = make_pdb_ctx_from_design(merged_s, torsion_angles, backbone_angles_sin_cos, backbone_affine_tensor)
                        ##########################################################################
                        absl_nat_pdb_f = f'{subdir}/{data_name}_{index}.pdb'
                        #############################################################################
                        prot = from_pdb_string(gt_pdb_ctx)
                        # struct_metrics = get_violation_metrics(prot, clash_overlap_tolerance=1.0)
                        # gt_num_residue_violations = struct_metrics['num_residue_violations']
                        log_dict[data_name]['comment'] = log_dict[f'{data_name}']['comment']

                        if args.write_allatom_model:
                            with open(absl_nat_pdb_f, 'w') as writer:
                                writer.write(gt_pdb_ctx)

                            torsion_continuous = design_ctx['torsion_continuous']
                            last_iter = torsion_continuous.shape[0] - 1
                            n_batch = iter_seq_list.shape[1]
                            for b_idx in range(n_batch):
                                global_b_idx = index * n_batch + b_idx
                                pdb_ctx = make_pdb_ctx_from_design(
                                    iter_seq_list[last_iter], torsion_continuous[last_iter].cpu(),
                                    backbone_angles_sin_cos.cpu(), backbone_affine_tensor.cpu(), b_idx)
                                absl_design_pdb_f = f'{subdir}/{data_name}_design_{global_b_idx}_{last_iter}.pdb'
                                with open(absl_design_pdb_f, 'w') as writer:
                                    writer.write(pdb_ctx)

                            # logger.info(f'all atoms model has been saved in {subdir} ')
                            #书写配体文件
                            # for ligname in features['lignames']:
                            #     mol = lig_raw_dict[ligname]['rdkit_ligand']
                            #     Chem.MolToMolFile(mol, f'{subdir}/{ligname}')


                        #保存fasta文件
                        ####################################################################################
                        #W是写入，并且将原来的内容清空
                        with open(f'{subdir}/{data_name}_{index}_design.fa', 'w') as writer:
                        #############################################################################
                            for protein, prot_ctx in log_dict.items():
                                writer.write('\n'.join([prot_ctx['comment'], prot_ctx['seq']]) + '\n')
                            logging.info(f" fasta file has been saved in {subdir}/{data_name}_{index}_design.fa")

                        # write_mem_position(design_ctx, subdir )

                        logger.info(f'{data_name} design finished, mean_val = {mean_val}, var_val = {var_val}')
                except Exception as e:
                    import traceback

                    traceback.print_exception(type(e), e, e.__traceback__)
                    if 'out of memory' in str(e).lower():
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
