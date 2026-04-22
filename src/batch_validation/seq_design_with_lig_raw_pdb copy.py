import os
import sys
import subprocess
import argparse
import json
from tqdm import tqdm
import math
from pathlib import Path
from copy import deepcopy
import tempfile
import ml_collections as mlc
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import torch
import torch.nn.functional as F
# from contextlib import contextmanager
sys.path.append("/home/chenty/abacust_mem/src/")

from rdkit import Chem
from fasde.models.diff_full_atom import DiffFullAtom
from fasde.modules.design_utils import  make_pdb_ctx_from_design

from fasde.data.utils.diffusion_utils_atoms import make_complex_feature, make_ligand_node_feature
from fasde.data.utils import features,target_features
from data_utils import pad_and_stack
from fasde.utils.relax.assess_violation import get_violation_metrics
from fasde.modules.alphafold.common.protein import from_pdb_string
from fairseq.data import data_utils
import traceback
sys.path.append('/home/chenty/abacust_mem/src/fasde/data/utils')
sys.path.append('/home/chenty/abacust_mem/src/fasde/data')
sys.path.append('/home/chenty/abacust_mem/src')
try:
    from .crop import Ellipsoid_crop
except:
    from crop import Ellipsoid_crop

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

    parser.add_argument('--device', type=str, default='cuda',help='cpu / cuda / cuda:0 / cuda:1 …')
    parser.add_argument("--tm_raw", type=str,default='raw')
    parser.add_argument("--data_name", type=str, default='')


    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    logger.info(f'checkpoint loaded: {checkpoint_path}')
    # return ckpt['last_optimizer_state']['state'][0]['step']

############################################################################################################################################
UNIMOL_REPRS_DIM=512
class FullAtomDataset():
    def __init__(
        self,
        seed,
        data_path,
        pdbtm_file_path, #pdbtm数据集生成npy文件，最重要的信息是膜的位置以及label标注
        pdb_path,       #pdb文件的路径（好像也用不上了）
        npy_path,   #pdb的npy的路径，主体
        split,
        config = None,
        mode="train",
        crop_size = 512,
        resize_len = False,
        lig_single_dim = 199,
        lig_pair_dim = 4,
        num_aatypes = 22,
        rec_beta_min = 0.1,
        rec_beta_max = 8,
        lig_beta_min = 0.1,
        lig_beta_max = 20,
        lig_center_mask_p = 0.6,
        fix_receptor_backbone = True,
        ca_radius=9.0,
        lig_neighbor_seq_mask=False,
        embed_unimol_reprs=False
    ):
        super().__init__()

        self.pdb_path = pdb_path
        self.max_len = crop_size
        self.resize_len = resize_len
        self.ca_radius = ca_radius
        self.lig_center_mask_p = lig_center_mask_p
        if lig_neighbor_seq_mask:
            merged_cluster_dict_f = '/home/chenty/abacust_mem/src/fasde/data/pdbs/merged_cluster_dict.npy'
        else:
            merged_cluster_dict_f = '/home/chenty/abacust_mem/src/fasde/data/pdbs/merged_cluster_dict.npy'


        self.pdbtm_file_path = pdbtm_file_path
        self.pdb_path = pdb_path
        self.npy_path = npy_path

        assert split in ['train', 'valid'], f"Invalid split value: {split}. Expected 'train' or 'valid'."
        try:
            self.split = split if split is not None else 'train'
        except Exception as e:
            import traceback;traceback.print_exc()
            import pdb;pdb.set_trace()


        self.merged_cluster_dict = np.load(merged_cluster_dict_f, allow_pickle=True).item()
        self.lig_single_dim = lig_single_dim
        self.lig_pair_dim = lig_pair_dim
        self.num_aatypes = num_aatypes
        self.embed_unimol_reprs = embed_unimol_reprs
        self.max_lig_num = 10
        self.pick_lig_p = 0.9

        ################################### plip annotation ###################################
        # plip_ligatom_npy = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/cry_all_prot_ligatom_pair_plip_anno_update.npy'
        # self.plip_ligatom_anno_dict = np.load(plip_ligatom_npy, allow_pickle=True).item()
        # self.plip_liatom_dropout_alllig_rate = 0.5
        ################################### plip annotation ###################################

        # self.merged_cluster_center_list = self.read_cluster_center_from_txt(data_path)
        # self.data_len = len(self.merged_cluster_center_list)

        self.merged_cluster_center_list = self.read_cluster_center_from_npy(merged_cluster_dict_f, split=self.split)

        self.data_len = len(self.merged_cluster_center_list)

        self.seed = seed
        self.mode = mode

        self.set_epoch(1)

    def get_data_index(self):
        return self.data_list

    def set_epoch(self, epoch):
        self.epoch = epoch

    def norm_coords(self, data):
        """进行坐标的归一化，找到所有ca原子的中心点，之后将所有原子进行整体的平移"""
        coords = data['coords']['multichain_merged_all_coords']
        coords = coords - coords[:, 1].mean(0)[None, None]
        data['coords']['multichain_merged_all_coords'] = coords
        return data



    def get_lig_ctx(self, lig_dict, features, consider_lig=True, max_lig_num=3):
        # logger.info(f"consider lig :{consider_lig}")
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

                # cur_lig_num = np.random.randint(1, max_lig_num+1)
                # sample_lig_num_p = (np.arange(max_lig_num)+1)/np.sum((np.arange(max_lig_num)+1))#计算概率密度，较多配体的权重更大
                # cur_lig_num = (np.random.choice(np.arange(max_lig_num), 1, replace=False, p=sample_lig_num_p) + 1)[0]#返回一个长度为1的一维数组，从np.arange(max_lig_num)里随机选择，不重复
                # cur_sel_lig_names = np.random.choice(lig_names, cur_lig_num, replace=False)#从lignames中随机选取cur_lig_num个进行分析，这是在给lig分析诸如随机性（真的可以吗）

                cur_sel_lig_names = lig_names
                lig_atom_num = 0
                lig_coords_list = []
                lig_single_list = []
                lig_single_unimol_list = []
                lig_edge_index_list = []
                lig_edge_attr_list = []
                ################################### plip annotation ###################################
                lig_plip_anno_itype_list = []
                ################################### plip annotation ###################################
                for lig_name in cur_sel_lig_names:
                    raw_data = lig_dict[lig_name]['data']
                    unimol_reprs = lig_dict[lig_name]['unimol_reprs']['atomic_reprs'][0]
                    cur_lig_atoms_num = unimol_reprs.shape[0]
                    lig_single = make_ligand_node_feature(raw_data['ligand']['x'])
                    lig_unimol_single = torch.from_numpy(unimol_reprs)

                    lig_edge = raw_data['ligand', 'lig_bond', 'ligand']
                    lig_edge_index = lig_edge.edge_index + lig_atom_num#原来的edgeindex加上之前所有配体累计的原子序号,形状为[B,2,M]
                    lig_edge_attr = lig_edge.edge_attr#链的特征，形状为[B,M,D]

                    lig_single_list.append(lig_single)
                    lig_single_unimol_list.append(lig_unimol_single)
                    lig_edge_index_list.append(lig_edge_index)
                    lig_edge_attr_list.append(lig_edge_attr)
                    lig_coords_list.append(raw_data['ligand']['pos'])

                    ################################### plip annotation ###################################
                    # cur_lig_anno_itype = torch.ones(cur_lig_atoms_num)
                    # query_lig_name = '_'.join(lig_name.split('.')[0].split('_')[:-1])
                    # if self.plip_ligatom_anno_dict.__contains__(features['pdbname'][0] + '_1_cry'):
                    #     cur_pdb_ligatom_plip_anno = self.plip_ligatom_anno_dict[features['pdbname'][0] + '_1_cry']
                    #     if cur_pdb_ligatom_plip_anno.__contains__(query_lig_name):
                    #         cur_lig_atom_plip_anno_dict = cur_pdb_ligatom_plip_anno[query_lig_name]
                    #         if (np.random.rand(1)[0] > self.plip_liatom_dropout_alllig_rate):
                    #             for plip_type, pres_latom_dict in cur_lig_atom_plip_anno_dict.items():
                    #                 plip_type_embed_id = plip_ligatom_anno[plip_type]
                    #                 if (np.random.rand(1)[0] < plip_liatom_dropout_ligatom_rate[plip_type]):
                    #                     continue
                    #                 for chainres, atom_raw_id_lst in pres_latom_dict.items():
                    #                     for atom_raw_id in atom_raw_id_lst:
                    #                         cur_lig_anno_itype[atom_raw_id-1] = plip_type_embed_id
                    # lig_plip_anno_itype_list.append(cur_lig_anno_itype)
                    ################################### plip annotation ###################################
                    lig_atom_num += cur_lig_atoms_num

                ################################### plip annotation ###################################
                # if len(lig_plip_anno_itype_list) > 0:
                #     # plip_anno_itype_list = torch.cat([torch.zeros(prot_len), torch.cat(lig_plip_anno_itype_list)], dim=0)
                #     plip_anno_itype_list = torch.zeros_like(torch.cat([torch.zeros(prot_len), torch.Tensor([0]*lig_atom_num)], dim=0))
                # else:
                #     plip_anno_itype_list = torch.zeros(prot_len)
                ################################### plip annotation ###################################

                lig_single_list = torch.cat(lig_single_list)
                lig_single_unimol_list = torch.cat(lig_single_unimol_list)
                lig_edge_index_list = torch.cat(lig_edge_index_list, -1)
                lig_edge_attr_list = torch.cat(lig_edge_attr_list)
                lig_coords_list = torch.cat(lig_coords_list)

                lig_atom_positions = torch.zeros((lig_atom_num, 4, 3))
                lig_atom_positions[:, 1] = lig_coords_list#最后的配体是存在第二行的，并且每一列是一个原子（与蛋白质部分不同，每一列是一个氨基酸）
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

                residx = torch.cat([prot_residx, torch.arange(lig_atom_num ) + prot_residx.max() + 100], dim=0)
                # residx_=(residx - torch.min(residx) + 1).tolist()
                residx_ = residx - torch.min(residx) + 1
                residx=residx_.clone().detach()#如果考虑配体，residx的结构是：从1开始依次递增，之后concat上配体的原子idx且中间间隔100
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
                residx = prot_residx - torch.min(prot_residx) + 1 #确保这里面最小的值是1
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
            residx = prot_residx - torch.min(prot_residx) + 1 #确保蛋白质的氨基酸都从1开始
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

        # 如果不考虑配体的话，以下变量都是直接抄下来就可以的
        lig_prot_features['tokens'] =  tokens
        lig_prot_features['coords'] =  all_atom_positions
        lig_prot_features['coords37'] = coords37
        lig_prot_features['residx'] =  residx
        lig_prot_features['node_mask'] =  node_mask #继承自prot_mask
        lig_prot_features['pdbname'] =  features['pdbname']
        lig_prot_features['prot_len'] = len(features['tokens'])
        lig_prot_features['total_len'] = len(tokens)


        lig_prot_features['lig_mask'] =  lig_mask #torch.zeros((prot_len,))prot_coords.shape[0]

        #如果不考虑配体，这两个量是全零
        lig_prot_features['lig_node_attr'] =  merged_lig_single
        lig_prot_features['unimol_lig_node_attr'] =  merged_lig_single_unimol#

        #配体的边索引和边特征
        lig_prot_features['lig_edge_index'] =  merged_lig_edge_index.transpose(0, 1)
        lig_prot_features['lig_edge_attr'] =  merged_lig_edge_attr

        lig_prot_features['chainidx'] =  chainidx#这个是将配体标记为1，链从2开始标记


        lig_prot_features['lignames'] =  cur_sel_lig_names
        plip_anno_itype_list = torch.zeros(tokens.shape[0])
        lig_prot_features['plip_anno_itype_list'] = plip_anno_itype_list


        lig_prot_features['chi_angles'] = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
        lig_prot_features['alt_chi_angles'] = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
        lig_prot_features['chi_mask'] = chi_mask
        lig_prot_features['backbone_affine_tensor'] = backbone_affine_tensor
        lig_prot_features['backbone_angles_sin_cos'] = backbone_angles_sin_cos
        lig_prot_features['lignames'] = cur_sel_lig_names


        return lig_prot_features

    def check_align(self,lig_prot_features):
        """从lig_prot_features中获取相应的氨基酸序列，链索引，pdbtm区域索引，氨基酸idx，原子坐标，
        如果不完全一样则返回false

        Args:
            lig_prot_features (_type_):输入的张量

        Returns:
            _type_: true/false
        """
        chainidx = lig_prot_features['chainidx']
        tokens = lig_prot_features['tokens']
        pdbtm_reigions = lig_prot_features['pdbtm_regions']
        res_idx = lig_prot_features['residx']
        all_atom_positions = lig_prot_features['coords']

        len_chain_idx = len(chainidx)
        len_tokens = len(tokens)
        len_all_atom_positions = len(all_atom_positions)
        len_res_idx = len(res_idx)
        len_pdbtm_regions = len(pdbtm_reigions)

        try:
            assert len_chain_idx == len_tokens == len_all_atom_positions == len_res_idx == len_pdbtm_regions, (
                    f"Lengths are not equal:\n"
                    f"len_chain_idx: {len_chain_idx}\n"
                    f"len_tokens: {len_tokens}\n"
                    f"len_all_atom_positions: {len_all_atom_positions}\n"
                    f"len_res_idx: {len_res_idx}\n"
                    f"len_pdbtm_regions: {len_pdbtm_regions}"
                )
            return True
        except Exception as e:
            traceback.print_exc(e)
            return False

    def read_cluster_center_from_txt(self,file_path):
        """
        从文本文件中读取键并返回列表。

        Args:
            file_path (str): 文本文件的路径。

        Returns:
            list: 包含文本文件中所有键的列表，或 None 表示失败。
        """
        try:
            with open(file_path, "r") as txt_file:
                keys = [line.strip() for line in txt_file.readlines()]
            print(f"Keys loaded from text file: {file_path}")
            return keys
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None
    def read_cluster_center_from_npy(self,file_path,split = "train"):
        """从cluster中取出train或者valid部分的keys(聚类中心)"""
        try:
            merged_cluster_dict = np.load(file_path, allow_pickle=True).item()
            sub_dict = merged_cluster_dict.get(split)
            keys = [str(k) for k in sub_dict.keys()]
            print(f"Keys loaded from npy file: {split}::{file_path} -> {len(keys)} keys")
            return keys
        except Exception as e:
            print(f"Error reading npy file: {e}")
            return None


    def __getitem__(self, data_name, strict_mode = True, crop = True):
        """fairseq会自动调用这个函数结合collator来输入样本，

        Args:
            idx (_type_): 想要获取的序列
            strict_mode (bool, optional): 如果为true则会检查输入的pdbtm张量以及pdb张量的tokens以及chains部分是否一致. Defaults to True.

        Returns:
            _type_: 单个样本字典，如果crop = true的话会对整个字典进行剪切
        """
        if self.split == 'valid':
            logger.info("validation epoch, using valid dataset")
        # rep_data_name = self.merged_cluster_center_list[(idx % len(self.merged_cluster_center_list))]        #self.data_list记录着所有的pdbid名字
        # rep_data_name = "6luq"
            # merged_cluster_dict = {
                #     'train':{
                #     rep_data_name_1:  [data_name_1, data_name_2, ..., data_name_n]
                #     ...
                # }
                # }
        try:
            # with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
            #     data_name = np.random.choice(self.merged_cluster_dict[self.split][rep_data_name])
            # data_name = '2rh1'
            print(data_name)
            consider_lig = True

            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + 42):
                # prot_lig_data_dict 示例格式：{
                #     'prot': {
                #         'atom_bb': np.ndarray,  # 主链原子坐标，形状为 (M, L, N, 3)
                #         'atom_37': np.ndarray,  # 37 个常见原子坐标，形状为 (M, L, N, 3)
                #         'pdbres_idx': np.ndarray,  # 残基索引信息
                #         'sequence': str  # 氨基酸序列
                #     },
                #     'lig': list  # 配体相关数据列表
                # }
                pdbcode = data_name  # 直接使用四位数的 PDB ID
                prot_lig_data_f = os.path.join(self.npy_path, f"{pdbcode}_complex.npy")
                prot_lig_data_dict = np.load(prot_lig_data_f, allow_pickle=True).item()

                prot_data_dict = prot_lig_data_dict['prot']
                atom_bb = prot_data_dict['atom_bb']
                atom_37 = prot_data_dict['atom_37']
                pdbresidx = prot_data_dict['pdbres_idx']  #  pdbres_idx 是原来pdb中的序列idx

                sequence = prot_data_dict['sequence']
                prot_len =  prot_data_dict['pdbres_idx_len']
                pdb_chain_mask = prot_data_dict['pdb_chain_mask']
                pdb_chain_mask = encode_protein_chain_mask(pdb_chain_mask, embedding_dict= chain_id_to_int,one_hot=False)

                atom_bb = torch.from_numpy(atom_bb).float() # M, L, N, 3
                atom_37 = torch.from_numpy(atom_37).float() # M, L, N, 3
                atom_37_mask = (~torch.all(atom_37 == 0, -1)).float()
                pdbresidx = torch.tensor(pdbresidx).long()
                node_mask = torch.ones(( prot_len,)).float()
                af2_tokens = torch.tensor([raw_restype_order[aatype_str] for aatype_str in sequence]).long()
                tokens = af2_tokens + 4
                # print(f"af2_tokens:{af2_tokens}")
                frame_results = target_features.get_rigid_groups(af2_tokens, atom_37, atom_37_mask)

                if (prot_len < 15):
                    return None

                features = {}

                features['coords'] = atom_bb[:, [0, 1, 2, 4]]#张量的维度没有变，形状变为（M,L,4,3）ca应该是第二列（？
                features['coords37'] = atom_37
                features['tokens'] = tokens
                features['node_mask'] = node_mask#将所有蛋白质部分标记为1
                features['residx'] = pdbresidx
                features['chainidx'] = pdb_chain_mask.long()  # 这里可以设置为固定的链索引，比如全部为 0
                features['pdbname'] = [pdbcode]  # 直接使用 PDB ID
                features.update(frame_results)

            # import pdb;pdb.set_trace()
            assert len(self.pdbtm_file_path) > 0
            pdbtm_file_path = os.path.join(self.pdbtm_file_path,f'{pdbcode}_minfo.npy')
            try:
                data = np.load(pdbtm_file_path, allow_pickle=True)
            except:
                raise ValueError(f'{pdbcode}_minfo.npy not exist!')


            #先处理膜矩阵信息
            attributes = data.item().get('attributes', {})
            membrane = attributes.get('MEMBRANE', {})

            normal_dict = membrane.get('NORMAL', {})
            features['normal'] = torch.tensor([float(value) for value in normal_dict.values()])
            features['tmatrix'] = torch.tensor(membrane.get('TMATRIX', np.array([])))  # 或者 torch.empty(0)

            biomarix_arr =  membrane.get('BIOMATRIX', np.array([]))
            if biomarix_arr is not None:
                features['biomatrix'] = torch.tensor(biomarix_arr)
            else:
                features['biomatrix'] = torch.empty(0)

            #之后处理跨膜类型的信息（目前有bug）
            # tp = attributes.get('RAWRES', {})
            # if tp:
            #     features['tmtype'] = tmtype_dict.get((tp[0].get('type', 'Unknown')), -1)
            # else:
            #     features['tmtype'] = tmtype_dict.get('Unknown', -1)

            #处理regions，seq,chain信息
            chains = data.item().get('chains', [])
            chains.sort(key=lambda x: x['chain_id'])

            pdbtm_tensor_list = []
            for chain in chains:
                warningtype = 0
                try:
                    seq_str = chain['sequence']
                    regions = chain['regions']
                    chain_idx = chain['chain_id']

                    pdbtm_region_list = []

                    for i,region in enumerate(regions):

                        try:
                            if region[0] in [ 'U','0']:
                                continue

                            #拆包，获取每一条链对应的seqidx和pdbidx
                            seq_start, seq_end = region[1]
                            pdb_start, pdb_end = region[2]

                            # 提取子序列
                            sub_sequence = seq_str[seq_start-1:seq_end]#算到seq_start-1位，如果这里是1，则是从第二位开始取
                            # sub_sequence = torch.tensor([raw_restype_order[aatype_str] for aatype_str in sub_sequence]).long() + 4#序列
                            sub_sequence = torch.tensor([raw_restype_order.get(s, 0) for s in sub_sequence]).long() + 4
                            sub_idx = torch.from_numpy(np.arange(seq_start, seq_end+1))#序列idx，从1开始
                            pdb_sub_idx = torch.from_numpy(np.arange(pdb_start, pdb_end+1))#序列idx#pdbidx


                            if pdb_end>=pdb_start:
                                region_tensor = torch.full(((pdb_end - pdb_start+1),), region_dict[region[0]])#位置标注掩码
                                chain_tensor = torch.full(((pdb_end - pdb_start+1),), (get_chain_id_value(chain_idx) + 2))#位置标注掩码,a链从第二个开始\
                            else:
                                #在原始的pdb文件中，可能会有一段氨基酸在pdb中的序列突然增加一段距离之后又回来
                                # 例如401，102，1003.....1050，451...
                                #这种情况放弃以pdb作为参照，转而以seq的idx作为参照
                                #但最终肯定还是要以pdb的序号为准，要不然和另一部分的npy对不齐
                                #只是将regiontensor和chaintensor相应的量做的长度一致而已
                                assert seq_end-seq_start>=0,f"{pdbcode}'s {chain['chain_id']}'s region's seq invalid!"

                                region_tensor = torch.full(((seq_end - seq_start+1),), region_dict[region[0]])
                                chain_tensor = torch.full(((seq_end - seq_start+1),), (get_chain_id_value(chain_idx) + 2))

                            tensors_to_stack = [sub_idx, sub_sequence, pdb_sub_idx, region_tensor, chain_tensor]
                            lengths = [t.size(0) for t in tensors_to_stack]
                            #确保都是等长的张量，因为有一些蛋白seq和pdbseq是对不齐的
                            if len(set(lengths)) != 1:
                                warningtype = 2
                                continue

                            # 将子序列转换为张量
                            try:
                                result_tensor = torch.stack(tensors_to_stack, dim=1)
                                pdbtm_region_list.append(result_tensor)
                            except Exception as e:
                                print([sub_idx, sub_sequence, pdb_sub_idx, region_tensor, chain_tensor])
                                traceback.print_exc(e)
                                import pdb;pdb.set_trace()


                        except Exception as e :
                            traceback.print_exc(e)
                            import pdb;pdb.set_trace()

                    if len(pdbtm_region_list) == 0:
                        warningtype = 1
                        continue
                        # import pdb;pdb.set_trace()
                    else:
                        pdbtm_region_list_tensor = torch.cat(pdbtm_region_list, dim=0)
                        pdbtm_tensor_list.append(pdbtm_region_list_tensor)
                        #组成以列为单位的张量列表，一共有五个张量：
                        #[sub_idx, ：从pdbtm的xml中获取的序列idx
                        # sub_sequence, ：从pdbtm的xml中获取的序列（已经编码之后+4）
                        # pdb_sub_idx, ：从pdbtm的xml中获取的pdb的idx，用来与pdb的data对齐
                        # region_tensor, ：region的编码
                        # chain_tensor ：链的标记（大概率与pdb的链一致，但有的时候会有个别增删）


                except Exception as e:
                    traceback.print_exc(e)
                    import pdb;pdb.set_trace()
                if warningtype == 1:
                    logger.warning(f"{features['pdbname']}'s {chain['chain_id']} pdbtm_list is empty")
                elif warningtype == 2:
                    logger.warning(f"{features['pdbname']}'s {chain['chain_id']} seq not align")


            pdbtm_tensor_list_tensor = torch.cat(pdbtm_tensor_list,dim=0)

            #pdb_tensor是完备的，序列连续，且有gap记录，chain递增且连续
            pdb_tensor_list = [
                features['chainidx'] + 2,#使得chainidx对齐
                features['residx'],
                features['tokens'],
            ]
            try:
                pdb_tensor_list = torch.stack(pdb_tensor_list,dim = 1)
            except:
                import pdb;pdb.set_trace()

            modefied_pdb_tensor,invalid_prot_mask = align_pdb_pdbtm_tensors(pdb_tensor_list,pdbtm_tensor_list_tensor,features['pdbname'])

            # import pdb;pdb.set_trace()
            modefied_pdb_tensor = modefied_pdb_tensor.long()
            features['residx'] = modefied_pdb_tensor[:,1]#好像还不是从1开始的吧
            features['pdbtm_regions'] = modefied_pdb_tensor[:, 3]
            features['pdbtm_regions_len'] = len(features['pdbtm_regions'])
            ###########################################################################################
            #仅保留pdbtm里面有的链
            keep_idx = (invalid_prot_mask == 0).nonzero(as_tuple=True)[0]  # 保留的索引
            for k, v in features.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == invalid_prot_mask.shape[0]:
                    features[k] = v.index_select(0, keep_idx)


            if strict_mode == True:
                if not self.check_align(features):
                    logger.warning("lig_prot_features not align!")
                    import pdb;pdb.set_trace()

            char2weight = {k: 1 if k in {'H','B','I','C','L'} else 0 for k in region_dict.keys()}
            # 2. 把映射表转成 tensor，方便一次性索引，假设字符的整数编码就是 region_dict 的值（0~9）
            weight_table = torch.tensor(
                    [char2weight[reverse_region_dict[i]] for i in range(len(region_dict))],
                    dtype=torch.float32
                )
            tm_region_mask = weight_table[features['pdbtm_regions']]
            # import pdb;pdb.set_trace()
            boundary_mask_left = (tm_region_mask.long()[2:] != tm_region_mask.long()[:-2])  # 左侧交界
            boundary_mask_right = (tm_region_mask.long()[:-2] != tm_region_mask.long()[2:])  # 右侧交界


            edge_mem_mask = torch.zeros_like(tm_region_mask).long()
            edge_mem_mask[2:] = torch.where(boundary_mask_left, torch.tensor(1, device=edge_mem_mask.device), edge_mem_mask[2:])
            edge_mem_mask[:-2] = torch.where(boundary_mask_right, torch.tensor(1, device=edge_mem_mask.device), edge_mem_mask[:-2])

            features['edge_mem_mask'] = edge_mem_mask

            if crop:
                features = crop_dict(features, threshold = self.max_len)#定义为256

            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + 42):
                if (len(prot_lig_data_dict['lig']) > 0):
                    if (np.random.rand(1)[0] < self.pick_lig_p):
                        consider_lig = True
                    else:
                        consider_lig = False
                # consider_lig = False
                lig_prot_features = self.get_lig_ctx(prot_lig_data_dict['lig'], features, consider_lig=consider_lig , max_lig_num=self.max_lig_num)
                # prot_lig_data_dict['lig'] = {
                #     'lig1': {
                #         'data': {
                #             'ligand': {
                #                 'x': torch.randint(0, 1, (4, 16)),  # 假设有4个配体原子特征维度为16
                #                 'pos': torch.randn(4, 3)  # 配体原子的3D坐标
                #             },
                #             ('ligand', 'lig_bond', 'ligand'): Data(
                #                 edge_index=torch.randint(0, 4, (2, 5)),
                #                 edge_attr=torch.randn(5, 3)
                #             )
                #         },
                #         'unimol_reprs': {
                #             'atomic_reprs': [np.random.randint(low=0, high=10, size=(4, 512))]  # 配体的原子级表示，维度为3x10
                #         }
                #     },
                #
            lig_prot_features['pdbname'] = features['pdbname']
            lig_prot_features['normal'] = features['normal']
            lig_prot_features['tmatrix'] = features['tmatrix']
            lig_prot_features['biomatrix'] = features['biomatrix']
            # lig_prot_features['tmtype'] = features['tmtype']
            lig_prot_features['pdbtm_regions'] =  torch.nn.functional.pad(features['pdbtm_regions'], (0, (lig_prot_features['coords'].shape[0]-features['pdbtm_regions'].shape[0])))#默认第二个维度是L
            lig_prot_features['pdbtm_regions_len'] = features['pdbtm_regions_len']
            lig_prot_features['edge_mem_mask'] = torch.nn.functional.pad(features['edge_mem_mask'], (0, (lig_prot_features['coords'].shape[0]-features["edge_mem_mask"].shape[0])))#默认第二个维度是L
            lig_prot_features['lig_feat'] = prot_lig_data_dict['lig']
            print(lig_prot_features['lignames'])

            # lig_prot_features['lignames'] = prot_lig_data_dict['lig'].keys()
            return lig_prot_features

        except Exception as e:
            traceback.print_exc()
            return None


    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples, device=torch.device('cuda')):
        # import pdb;pdb.set_trace()
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None
        batch = {}
        max_len = max([sample['coords'].shape[0] for sample in samples])
        #将所有类型的量从每个样本为一级目录变为每个数据类型为一级目录，
        for k in samples[0].keys():
            if k in ['tokens']:
                batch[k] = pad_and_stack_([s[k] for s in samples], batch_first=True, value=1).to(device, non_blocking=True)
            elif k in ['lig_neighbor_seq_mask']:
                batch[k] = pad_and_stack_([s[k] for s in samples], batch_first=True, value=True).to(device, non_blocking=True)
            elif k in ['lig_edge_index']:
                batch[k] = pad_and_stack_([s[k] for s in samples], batch_first=True, value=-2).to(device, non_blocking=True)
            elif k in ['pdbname', 'lignames','tmtype','rec_len']:
                batch[k] = [s[k] for s in samples]
            elif k in ['lig_feat']:
                pass
            else:
                try:
                    batch[k] = pad_and_stack_([s[k] for s in samples], batch_first=True, value=0).to(device, non_blocking=True)
                except Exception as e:
                    traceback.print_exc()
                    import pdb;pdb.set_trace()
        return batch

    def num_tokens(self, index):
        return self.max_len

    def size(self, index):
        return self.num_tokens(index)

def to_tensor(arr):
    if arr.dtype in [np.int64, np.int32]:
        return torch.LongTensor(arr)
    elif arr.dtype in [np.float64, np.float32]:
        return torch.FloatTensor(arr)
    elif arr.dtype == np.bool_:
        return torch.BoolTensor(arr)
    else:
        return arr

def align_pdb_pdbtm_tensors(pdb_tensor, pdbtm_tensor,pdbname = None):
    """
    将 PDB 和 PDBTM 张量对齐，根据链进行分组，并将 PDBTM 的区域信息整合到 PDB 张量中。

    Args:
        pdb_tensor (torch.Tensor): PDB 数据张量，pdb_tensor_list = [
                features['chainidx'],
                features['residx'],
                features['tokens'],
            ]
        pdbtm_tensor (torch.Tensor): PDBTM 数据张量， [sub_idx, sub_sequence, pdb_sub_idx, region_tensor, chain_tensor]

    Returns:
        torch.Tensor: 整合了 PDBTM 区域信息的 PDB 张量。
    """

    # 把U和0洗掉先，要不然甚至有可能对不齐
    # chain_mask = pdbtm_tensor[:, 4]
    # cols_to_keep = (chain_mask != region_dict['0']) & (chain_mask != region_dict['U'])
    # pdbtm_tensor = pdbtm_tensor[:, cols_to_keep]

    # 按链对 PDBTM 张量进行分组
    tm_unique_keys = torch.unique(pdbtm_tensor[:, 4])  # 获取唯一的链标识符
    pdbtm_grouped = {key.item(): pdbtm_tensor[pdbtm_tensor[:, 4] == key] for key in tm_unique_keys}

    # 按链对 PDB 张量进行分组
    unique_keys = torch.unique(pdb_tensor[:, 0])
    grouped_pdb = {key.item(): pdb_tensor[pdb_tensor[:, 0] == key] for key in unique_keys}

    # 初始化一个列表来存储每个链的区域信息
    append_regions = []
    invalid_prot_mask = []
    place_for_ligs = []

    # import pdb;pdb.set_trace()


    # 遍历每个链的 PDB 张量
    for key, pdb_chain in grouped_pdb.items():
        if key == 1:  # 配体链直接跳过
            place_for_ligs.append(torch.zeros((pdb_chain.shape[0],)))#希望shaoe【0】指的是长度
            continue

        # 初始化一个全零的张量，用于标记区域



        if key in pdbtm_grouped:
            region_labels = torch.zeros(pdb_chain.shape[0])
            invalid_prot_labels = torch.zeros(pdb_chain.shape[0])
            # 如果当前链在 PDBTM 中存在对应的链,则将其写入并对齐，检查序列是否一致#考虑大小写
            tm_chain = pdbtm_grouped[key]

            # 遍历 PDB 链中的每个索引
            for i in range(pdb_chain.shape[0]):
                current_idx = pdb_chain[i, 1]#当前pdb张量中的idx，在第一行,current_idx指的是pdbchain中的residx部分中的第i个idx（pdb的）

                # 检查当前索引是否在 PDBTM 链中
                if current_idx in tm_chain[:, 2]:#pdbt中pdb的索引在第三行
                    # 找到对应的 PDBTM 索引位置
                    try:
                        tm_indices = torch.where(tm_chain[:, 2] == current_idx)[0].item()#tm_chain[:, 2]是pdbtm中的pdb——idx部分
                    except Exception as e:
                        traceback.print_exc()
                        # import pdb;pdb.set_trace()


                    # 验证序列是否相同
                    if tm_chain[tm_indices, 1] == pdb_chain[i, 2]:
                        region_labels[i] = tm_chain[tm_indices, 3]  # 区域信息存储在第 3 列
                    elif pdb_chain[i, 2].item() == 11:#有的时候pdb里面会有unk的氨基酸，表示不知道，在parse的时候可能直接变成甘氨酸了（11），就这样吧
                        region_labels[i] = tm_chain[tm_indices, 3]
                    else:

                        logger.warning(f"{pdbname}--{key}: token clash!")
                        region_labels[i] = tm_chain[tm_indices, 3]#如果出现冲突，报警告，照常加，

                else:
                    region_labels[i] = 0  # 索引不在 PDBTM 中，标记为 0

            append_regions.append(region_labels.long())
            invalid_prot_mask.append(invalid_prot_labels)# 0表示当前链不需要被舍弃

        else:
            # 当前链在 PDBTM 中不存在对应的链，全部标记为 0
            region_labels = torch.zeros(pdb_chain.shape[0])
            append_regions.append(region_labels.long())
            invalid_prot_mask.append(torch.ones(pdb_chain.shape[0]).long()) #全一表示之后需要去掉
            pass



    append_regions = append_regions + place_for_ligs
    # import pdb;pdb.set_trace()
    if len(place_for_ligs) > 0:
        invalid_prot_mask = invalid_prot_mask.append(torch.zeros_like(place_for_ligs[0]))


    # 将区域标签列表转换为张量
    append_regions_tensor = torch.cat(append_regions, dim=0).long()
    invalid_prot_mask_tensor = torch.cat(invalid_prot_mask,dim=0).long()

    # 将区域标签张量拼接到 PDB 张量上
    pdb_tensor = torch.cat((pdb_tensor, append_regions_tensor.view(-1, 1)), dim=1)


    return pdb_tensor,invalid_prot_mask_tensor


def pad_and_stack_(tensors, batch_first = True, value=0):
    """
    对一批数据进行填充和堆叠。数据可以是张量、整数或浮点数。
    根据 batch_first 决定堆叠发生在第一层还是第二层。

    Args:
        tensors (list): 要处理的数据列表，可以包含张量、整数或浮点数。
        batch_first (bool, optional): 如果为 True，输出张量的形状为 (batch_size, max_length)；
                                    如果为 False，输出张量的形状为 (max_length, batch_size)。
        value (int or float, optional): 填充值。默认为 0。

    Returns:
        torch.Tensor: 填充并堆叠后的张量。
    """
    converted_tensors = []
    for t in tensors:
        if isinstance(t, (int, float)):
            # 如果是整数或浮点数，转换为张量
            converted_tensors.append(torch.tensor([t]))
        elif isinstance(t, torch.Tensor):
            converted_tensors.append(t)
        else:
            raise TypeError(f"Unsupported type {type(t)} in the input list.")

    # 检查转换后的张量列表是否为空
    if not converted_tensors:
        return torch.empty(0)

    # 获取每个张量的长度
    lengths = [tensor.size(0) for tensor in converted_tensors]

    # 对张量进行填充
    padded_tensors = rnn_utils.pad_sequence(converted_tensors, batch_first=batch_first, padding_value=value)

    return padded_tensors

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


def crop_dict(lig_prot_features, threshold, eccentricity = 0.5, std_dev=15):
    if 'coords' not in lig_prot_features:
        print("'coords' 键不存在")
        return None

    # 获取点集
    points = lig_prot_features['coords'][:, 1, :]

    if not isinstance(points, torch.Tensor):
        print("'coord' 的值不是张量")
        return None

    # 检查张量形状
    if len(points.shape) != 2 or points.shape[1] != 3:
        print(f"张量形状不正确，应为 (N, 3),但实际是{points.shape}")
        return None

    # 检查点的数量是否超过阈值
    if points.shape[0] > threshold:
        # print(f"点的数量 {points.shape[0]} 超过阈值 {threshold}，进行裁剪")
        try:
            try:
                rotation_matrix = lig_prot_features['tmatrix'][:3, :3].numpy()
            except Exception as e:
                import pdb;pdb.set_trace()
            diagonal_matrix = np.diag([1, 1, 1/eccentricity])
            affine_matrix = np.dot(np.linalg.inv(diagonal_matrix), rotation_matrix)
            indices = Ellipsoid_crop.crop_and_search_return_idx(points, affine_matrix, threshold, std_dev)

            for k in lig_prot_features.keys():
                if k in ['tokens', 'coords', 'coords37' ,'backbone_angles_sin_cos',
                            'backbone_affine_tensor', 'chi_mask','chainidx',"residx",'node_mask','chi_angles','alt_chi_angles','pdbtm_regions','edge_mem_mask'
                            ]:
                    lig_prot_features[k] = lig_prot_features[k][indices]
                elif k in ['pdbname', 'lignames','rec_len', 'tmtype','tmatrix','biomatrix',
                            'prot_len','total_len','normal','pdbtm_regions_len','idx', 'lig_neighbor_seq_mask']:
                    pass
                elif k in ["torsion_angles_sin_cos"]:
                    pass
                elif k in ['lig_edge_index', "lig_edge_attr"]:#待会想想怎么搞这俩玩意
                    pass
                else:
                    pass
                    # logger.info(f"keys {k} not mentioned!")
                    # lig_prot_features[k] = lig_prot_features[k][indices]

        except Exception as e:
            traceback.print_exc()
            print(f"裁剪过程中发生错误: {e}")
            import pdb;pdb.set_trace()
            return None

    # 按照 chain 和 idx 进行排序
    try:
        chains = lig_prot_features.get('chainidx', None)
        idxs = lig_prot_features.get('residx', None)

        assert chains is not None and idxs is not None ,"chains and idxs are null in sorting"
        sorted_indices = np.lexsort((idxs.numpy(),chains.numpy()))
        keys_to_sort = ['tokens', 'coords', 'coords37' ,'backbone_angles_sin_cos',
                            'backbone_affine_tensor', 'chi_mask','chainidx',"residx",'node_mask','chi_angles','alt_chi_angles']
        for k in keys_to_sort:
            assert k in lig_prot_features
            lig_prot_features[k] = lig_prot_features[k][sorted_indices]
        return lig_prot_features

    except Exception as e:
        print(f"error in sorting: {e}")
        traceback.print_exc()
        return None
##################################################################################################################################


# def get_lig_ctx(lig_dict, features, consider_lig=True, max_lig_num=3, UNIMOL_REPRS_DIM=512):
#     lig_prot_features = {}
#     prot_coords = features['coords']
#     prot_coords37 = features['coords37']
#     prot_tokens = features['tokens']
#     prot_mask = features['node_mask']
#     prot_residx = features['residx']
#     prot_chainidx = features['chainidx']
#     prot_chi_angles = features['chi_angles']
#     prot_alt_chi_angles = features['alt_chi_angles']
#     prot_chi_mask = features['chi_mask']
#     prot_backbone_affine_tensor = features['backbone_affine_tensor']
#     prot_backbone_angles_sin_cos = features['backbone_angles_sin_cos']
#     prot_len = prot_coords.shape[0]

#     if consider_lig:
#         lig_names = list(lig_dict.keys())
#         avail_lig_num = len(lig_dict)

#         if (avail_lig_num > 0):
#             if (avail_lig_num < max_lig_num):
#                 max_lig_num = avail_lig_num

#             cur_sel_lig_names = np.random.choice(lig_names, max_lig_num, replace=False)
#             lig_atom_num = 0
#             lig_coords_list = []
#             lig_single_list = []
#             lig_single_unimol_list = []
#             lig_edge_index_list = []
#             lig_edge_attr_list = []

#             lig_plip_anno_itype_list = []

#             for lig_name in cur_sel_lig_names:
#                 raw_data = lig_dict[lig_name]['data']
#                 unimol_reprs = lig_dict[lig_name]['unimol_reprs']['atomic_reprs'][0]
#                 cur_lig_atoms_num = unimol_reprs.shape[0]
#                 lig_single = make_ligand_node_feature(raw_data['ligand']['x'])
#                 lig_unimol_single = torch.from_numpy(unimol_reprs)

#                 lig_edge = raw_data['ligand', 'lig_bond', 'ligand']
#                 lig_edge_index = lig_edge.edge_index + lig_atom_num
#                 lig_edge_attr = lig_edge.edge_attr

#                 lig_single_list.append(lig_single)
#                 lig_single_unimol_list.append(lig_unimol_single)
#                 lig_edge_index_list.append(lig_edge_index)
#                 lig_edge_attr_list.append(lig_edge_attr)
#                 lig_coords_list.append(raw_data['ligand']['pos'])

#                 cur_lig_anno_itype = torch.ones(cur_lig_atoms_num)
#                 lig_plip_anno_itype_list.append(cur_lig_anno_itype)

#                 lig_atom_num += cur_lig_atoms_num

#             plip_anno_itype_list = torch.cat([torch.zeros(prot_len), torch.cat(lig_plip_anno_itype_list)], 0)
#             lig_single_list = torch.cat(lig_single_list)
#             lig_single_unimol_list = torch.cat(lig_single_unimol_list)
#             lig_edge_index_list = torch.cat(lig_edge_index_list, -1)
#             lig_edge_attr_list = torch.cat(lig_edge_attr_list)
#             lig_coords_list = torch.cat(lig_coords_list)

#             lig_atom_positions = torch.zeros((lig_atom_num, 4, 3))
#             lig_atom_positions[:, 1] = lig_coords_list
#             all_atom_positions = torch.cat([prot_coords, lig_atom_positions], dim=0)
#             segment = torch.LongTensor([0] * prot_len + [1] * lig_atom_num)
#             frame_len = prot_len + lig_atom_num
#             ## single
#             rec_single = torch.ones((prot_len,1))
#             rec_single_dim = rec_single.shape[-1]
#             lig_single_dim = lig_single_list.shape[-1]
#             single = merged_lig_single = torch.zeros((frame_len, rec_single_dim + lig_single_dim), dtype=torch.float)
#             single[:prot_len, :rec_single_dim] = rec_single
#             single[prot_len:, rec_single_dim:] = lig_single_list
#             merged_lig_single[prot_len:, rec_single_dim:] = lig_single_list
#             single = torch.cat([single, F.one_hot(segment, num_classes=2)], dim=-1)
#             ## unimol single
#             merged_lig_single_unimol = torch.zeros((frame_len, UNIMOL_REPRS_DIM), dtype=torch.float)
#             merged_lig_single_unimol[prot_len:] = lig_single_unimol_list
#             ## prot edge
#             rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
#             st, ed = rec_edge_index[0], rec_edge_index[1]
#             is_covalent = (prot_residx[ed] - prot_residx[st]) == 1
#             rec_edge_index = rec_edge_index[:, is_covalent]
#             rev_edge_index = rec_edge_index.clone()
#             rev_edge_index[0] = rec_edge_index[1]
#             rev_edge_index[1] = rec_edge_index[0]
#             rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
#             num_rec_edge = rec_edge_index.shape[1]
#             ## lig edge
#             edge_index = torch.cat([rec_edge_index, lig_edge_index_list + prot_len], dim=1)
#             merged_lig_edge_index = torch.cat([torch.ones_like(rec_edge_index) * -2, lig_edge_index_list + prot_len], dim=1)
#             edge_attr = merged_lig_edge_attr = torch.zeros((edge_index.shape[1], lig_edge_attr.shape[1] +1))
#             edge_attr[:num_rec_edge, 0] = 1
#             edge_attr[num_rec_edge:, 1:] = lig_edge_attr_list
#             merged_lig_edge_attr[num_rec_edge:, 1:] = lig_edge_attr_list

#             logger.info("residx_changed")

#             residx = torch.cat([prot_residx, torch.arange(lig_atom_num ) + prot_residx.max() + 100], dim=0)
#             residx_=(residx - torch.min(residx) + 1).tolist()
#             residx=torch.tensor(residx_)
#             # residx_ = residx
#             # residx=residx_.clone().detach()
#             lig_mask = torch.ones((frame_len,))
#             lig_mask[:prot_len] = 0
#             tokens = torch.cat([ prot_tokens, torch.LongTensor([24]*lig_atom_num)], dim=0)
#             node_mask = torch.ones((frame_len,))
#             prot_node_mask = torch.ones((prot_len,))
#             chainidx = torch.zeros((frame_len,)).long()
#             chainidx[:prot_len] = prot_chainidx + 1
#             chainidx = chainidx + 1

#             pad_angles = torch.stack([torch.ones(lig_atom_num, 4), torch.zeros(lig_atom_num, 4)], -1)
#             chi_angles = torch.cat([prot_chi_angles, pad_angles], 0)
#             alt_chi_angles = torch.cat([prot_alt_chi_angles, pad_angles], 0)
#             chi_mask = torch.cat([prot_chi_mask, torch.zeros(lig_atom_num, 4)], 0)
#             backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor, torch.ones(lig_atom_num, 4, 12)], 0)
#             coords37 = torch.cat([prot_coords37, torch.zeros(lig_atom_num, 37, 3)])
#             fake_backbone_angles_sin_cos = torch.stack([torch.ones(lig_atom_num, 3), torch.zeros(lig_atom_num, 3)], -1)
#             backbone_angles_sin_cos = torch.cat([prot_backbone_angles_sin_cos, fake_backbone_angles_sin_cos], 0)

#         else:
#             plip_anno_itype_list = torch.zeros(prot_len)
#             all_atom_positions = prot_coords
#             residx = prot_residx - torch.min(prot_residx) + 1
#             # residx = prot_residx
#             node_mask = prot_mask
#             lig_mask = torch.zeros((prot_len,))
#             merged_lig_single = torch.zeros((prot_len, 200), dtype=torch.float)
#             merged_lig_single_unimol = torch.zeros((prot_len, UNIMOL_REPRS_DIM), dtype=torch.float)

#             rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
#             st, ed = rec_edge_index[0], rec_edge_index[1]
#             is_covalent = (residx[ed] - residx[st]) == 1
#             rec_edge_index = rec_edge_index[:, is_covalent]
#             rev_edge_index = rec_edge_index.clone()
#             rev_edge_index[0] = rec_edge_index[1]
#             rev_edge_index[1] = rec_edge_index[0]
#             rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
#             num_rec_edge = rec_edge_index.shape[1]

#             edge_index = rec_edge_index
#             merged_lig_edge_index = (torch.ones_like(rec_edge_index) * -2) #.transpose(0, 1)
#             merged_lig_edge_attr = torch.zeros((edge_index.shape[1], 5))

#             tokens = prot_tokens
#             chainidx = torch.zeros((prot_len,)).long()
#             chainidx[:prot_len] = prot_chainidx + 1
#             chainidx = chainidx + 1
#             cur_sel_lig_names = []

#             chi_angles = torch.cat([prot_chi_angles], 0)
#             alt_chi_angles = torch.cat([prot_alt_chi_angles], 0)
#             chi_mask = torch.cat([prot_chi_mask], 0)
#             backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor], 0)
#             coords37 = prot_coords37
#             backbone_angles_sin_cos = prot_backbone_angles_sin_cos

#     else:
#         plip_anno_itype_list = torch.zeros(prot_len)
#         all_atom_positions = prot_coords
#         # residx = prot_residx
#         residx = prot_residx - torch.min(prot_residx) + 1
#         node_mask = prot_mask
#         lig_mask = torch.zeros((prot_len,))
#         merged_lig_single = torch.zeros((prot_len, 200), dtype=torch.float)
#         merged_lig_single_unimol = torch.zeros((prot_len, UNIMOL_REPRS_DIM), dtype=torch.float)

#         rec_edge_index = torch.stack([torch.arange(prot_len-1), torch.arange(prot_len-1)+1], dim=0)
#         st, ed = rec_edge_index[0], rec_edge_index[1]
#         is_covalent = (residx[ed] - residx[st]) == 1
#         rec_edge_index = rec_edge_index[:, is_covalent]
#         rev_edge_index = rec_edge_index.clone()
#         rev_edge_index[0] = rec_edge_index[1]
#         rev_edge_index[1] = rec_edge_index[0]
#         rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
#         num_rec_edge = rec_edge_index.shape[1]

#         edge_index = rec_edge_index
#         merged_lig_edge_index = (torch.ones_like(rec_edge_index) * -2) #.transpose(0, 1)
#         merged_lig_edge_attr = torch.zeros((edge_index.shape[1], 5))

#         tokens = prot_tokens
#         chainidx = torch.zeros((prot_len,)).long()
#         chainidx[:prot_len] = prot_chainidx + 1
#         chainidx = chainidx + 1
#         cur_sel_lig_names = []

#         chi_angles = torch.cat([prot_chi_angles], 0)
#         alt_chi_angles = torch.cat([prot_alt_chi_angles], 0)
#         chi_mask = torch.cat([prot_chi_mask], 0)
#         backbone_affine_tensor = torch.cat([prot_backbone_affine_tensor], 0)
#         coords37 = prot_coords37
#         backbone_angles_sin_cos = prot_backbone_angles_sin_cos

#     lig_prot_features['tokens'] =  tokens
#     plip_anno_itype_list = torch.zeros(tokens.shape[0])
#     lig_prot_features['coords'] =  all_atom_positions
#     lig_prot_features['node_mask'] =  node_mask
#     lig_prot_features['lig_mask'] =  lig_mask
#     lig_prot_features['residx'] =  residx
#     lig_prot_features['lig_node_attr'] =  merged_lig_single
#     lig_prot_features['unimol_lig_node_attr'] =  merged_lig_single_unimol
#     lig_prot_features['lig_edge_index'] =  merged_lig_edge_index.transpose(0, 1)
#     lig_prot_features['lig_edge_attr'] =  merged_lig_edge_attr
#     lig_prot_features['chainidx'] =  chainidx
#     lig_prot_features['pdbname'] =  features['pdbname']
#     lig_prot_features['lignames'] =  cur_sel_lig_names
#     lig_prot_features['plip_anno_itype_list'] =  plip_anno_itype_list

#     lig_prot_features['chi_angles'] = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
#     lig_prot_features['alt_chi_angles'] = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
#     lig_prot_features['chi_mask'] = chi_mask
#     lig_prot_features['backbone_affine_tensor'] = backbone_affine_tensor
#     lig_prot_features['backbone_angles_sin_cos'] = backbone_angles_sin_cos
#     lig_prot_features['coords37'] = coords37

#     try:
#         lig_prot_features['lig_coords'] = lig_coords_list
#     except:
#         pass

#     return lig_prot_features



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
                import traceback;traceback.print_exc()
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
    try:
        base_architecture(args)
        device = torch.device(args.device)
        logger.info(f'Running on device: {device}')

        model = DiffFullAtom(args)
        load_checkpoint(model, args.checkpoint)
        model.to(device)
        model.eval()
        # model.train()

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
        max_lig_num = args.max_lig_num
        temperature = args.temperature
        batchsize = args.batchsize
        data_name_str = args.data_name
        data_name_list = [x.strip() for x in data_name_str.split(",")]

        if args.bg_dist_file is not None and args.bg_dist_key is not None:#背景文件（什么玩意？）
            bg_dict = np.load(args.bg_dist_file, allow_pickle=True).item()
            bg_dist = torch.softmax(bg_dict[args.bg_dist_key], -1)
        else:
            bg_dist = None
        # print("reached here######################################")
        with torch.no_grad():
            full_atom_dataset = FullAtomDataset(
                                                42,#seed
                                                None,
                                                "/home/chenty/abacust_mem/src/data/prot_lists/pdbtm_data/tm_npys/",
                                                None,
                                                "/home/chenty/abacust_mem/src/fasde/data/pdbs/all_npy",
                                                split="train",
                                                mode="train",
                                                crop_size =256,
                                                resize_len = False,
                                                lig_single_dim = 199,
                                                lig_pair_dim = 4,
                                                num_aatypes = 22,
                                                fix_receptor_backbone = True,
                                                ca_radius = 9.0,
                                                lig_neighbor_seq_mask=None,#这个是啥
                                                embed_unimol_reprs=True
                                                )
            # for data_name in data_name_list:
            for data_name in tqdm(data_name_list, desc="processing"):
                logger.info(f"processing {data_name}")
                features = full_atom_dataset.__getitem__(data_name, strict_mode = True, crop = False)
                print("#################features loaded!")

                lig_raw_dict = features['lig_feat']
                try:
                    batch = FullAtomDataset.collater([features],device=torch.device('cuda'))
                except Exception as e:
                    import traceback;traceback.print_exc()
                    continue


                # batch = collater([features], device = device)
                design_ctx = {}
                if batch['coords'].shape[1] >= 900:
                    logger.info(f"{data_name}'s length is over 900")
                    continue
                for cur_batchsize in range(batchsize, 0, -1):
                    try:
                        batch = expand_batch(batch, cur_batchsize)
                        # batch = expand_batch(batch, batchsize)#组合成一个批次

                        print('cur_batchsize =', cur_batchsize, 'batch coords shape =', batch['coords'].shape)

                        pdbtm_regions = batch['pdbtm_regions']
                        edge_mem_mask = batch['edge_mem_mask']
                        G = batch['tmatrix']
                        N = batch['normal']
                        pdbname = batch["pdbname"]


                        #这一段是原来design的部分，seq_mask_pocket只能影响到配体
                        merged_coords = batch['coords']
                        merged_s = batch['tokens']
                        merged_mask = batch['node_mask']

                        B, L = merged_coords.shape[:2]
                        merged_chain_mask = batch['chainidx']
                        merged_residue_idx = batch['residx']
                        # # print(batch["residx"])

                        if bg_dist is not None:
                            bg_dist_prob = torch.zeros((1, L, 35)).to(device)
                            for residx in range(bg_dist.shape[0]):
                                for aa in raw_restypes:
                                    bg_dist_prob[0, residx, restype_order[aa]] = bg_dist[residx, esm_dict[aa]]
                            bg_dist_prob = bg_dist_prob/(torch.sum(bg_dist_prob, -1, keepdim=True) + 1e-8)
                        else:
                            bg_dist_prob = None

                        merged_chain_encoding_all = batch['chainidx']
                        # # randn = torch.randn_like(merged_chain_mask)

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
                        # alt_chi_angles = batch['alt_chi_angles']
                        torsion_angle_mask=batch['chi_mask']
                        backbone_affine_tensor = batch['backbone_affine_tensor']
                        backbone_angles_sin_cos = batch['backbone_angles_sin_cos']
                        plip_anno_itype_list = batch['plip_anno_itype_list']

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
                        )
                        break
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            # import traceback;traceback.print_exc()
                            print(f"cuda still out of memory when batchsize is {cur_batchsize} !")
                            import time;time.sleep(1)
                            torch.cuda.empty_cache()
                            continue
                        raise



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
                # for iter_idx in return_pdb_ctx_iter:
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
                            # absl_design_pdb_f = f'{subdir}/{data_name}_design_{b_idx}_{iter_idx}.pdb'
                            # with open(absl_design_pdb_f, 'w') as writer:
                            #     writer.write(pdb_ctx)

                    logger.info(f'all atoms model has been saved in {subdir}')
                    #书写配体文件
                    print(features["lignames"])
                    for ligname in features['lignames']:
                        mol = lig_raw_dict[ligname]['rdkit_ligand']
                        Chem.MolToMolFile(mol, f'{subdir}/{ligname}')
                        logger.info(f'{ligname} has been added')
                #保存蛋白质氨基酸相对膜的位置
                # mem_position = design_ctx["mem_position"]
                # mem_position_np = mem_position.detach().cpu().numpy()
                # np.save(os.path.join(subdir, f'{data_name}_position.npy'), mem_position_np)

                #保存fasta文件
                with open(f'{subdir}/{data_name}_design.fa', 'w') as writer:
                    for protein, prot_ctx in log_dict.items():
                        writer.write('\n'.join([prot_ctx['comment'], prot_ctx['seq']]) + '\n')

                # 写入每次设计得均值方差
                with open(f'{subdir}/{data_name}_design.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{data_name}:mean_val={mean_val};var_val={var_val}\n')
                logger.info(f'{data_name} design finished')
                logger.info(f"mean_val = {mean_val}")
                logger.info(f"var_val = {var_val}")

                    # torch.cuda.empty_cache()


    except RuntimeError as e:
        import traceback;traceback.print_exception()

        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args()
    main(args)
