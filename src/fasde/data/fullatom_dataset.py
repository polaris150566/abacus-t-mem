import json
import traceback
import pickle
import ml_collections as mlc
import numpy as np
import logging
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from typing import *
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
from pathlib import Path


#数据处理，蛋白质npy文件中将所有的链拼接在一起，并且加上chainidx辅助辨认，
#pdbtm的regions也拼接在一起，并且将chain也加上，之后将两条链进行对比
#chain的编码方式：字母到数字的直接映射，类型：张量

sys.path.append('/home/chenty/abacust_mem/src/fasde/data/utils')
sys.path.append('/home/chenty/abacust_mem/src')
#from #原版代码里面有。utils
#
try:
    from .utils.diffusion_utils_atoms import make_ligand_node_feature, restype_order
    from .utils import features,target_features
except:
    import features,target_features
    from diffusion_utils_atoms import make_ligand_node_feature #, restype_order

from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
import os
try:
    from .crop import Ellipsoid_crop
except:
    from crop import Ellipsoid_crop

# path='/home/chenty/abacust_mem/src/fasde/data/config.json'
# data_config=mlc.ConfigDict(json.loads(open(path).read()))

UNIMOL_REPRS_DIM=512

logger = logging.getLogger(__file__)
logger.name = 'fullatom_dataset'




plip_ligatom_anno = {
    'mask': 0, 'unknown': 1, 'pistacking': 2, 'pication_laro': 3, 'pication_paro': 4, 'hydrophobic': 5, 'hbond_ldon': 6, 'hbond_pdon': 7, 'metalcomplex': 8, 'saltbridge_lneg': 9, 'saltbridge_pneg': 10, 'halogen': 11, 'waterbridge': 12
}
plip_liatom_dropout_ligatom_rate = {'pistacking': 0.4, 'pication_laro': 0.4, 'pication_paro': 0.4, 'hydrophobic': 0.5, 'hbond_ldon': 0.2, 'hbond_pdon': 0.2, 'metalcomplex': 0.0, 'saltbridge_lneg': 0.2, 'saltbridge_pneg': 0.2, 'halogen': 0.0, 'waterbridge': 1.0}
restype_order = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
                    'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': -1, 'U': -2, '?':-3}#后3个是手动添加的
order_restype = {v: k for k, v in restype_order.items()}

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

#蛋白质链的映射方式:直接按照字母表的顺序进行映射
#在对于pdb文件进行chain的编码时，需要额外加二，因为0不被考虑，而1表示配体，所以A链是从2开始的
chain_id_to_int = {chr(ord('A') + i): i for i in range(26)}
chain_id_to_int.update({chr(ord('a') + i): 26 + i for i in range(26)})#考虑小写字母
chain_id_to_int.update({str(i): 52 + i for i in range(20)})#数字编号也加上
def get_chain_id_value(chain_id):
    return chain_id_to_int.get(chain_id, -1)

tmtype_dict = {
    "Unknown":-1, "Soluble": 0, "No_Protein": 1, "Nucleotide": 2, "Ca_Globular": 3, "Virus": 4,
    "Pilus": 5, "Tm_Alpha": 6, "Tm_Beta": 7, "Tm_Coil": 8, "Ca_Tm": 9, "Tm_Part": 10
}



class FullAtomDataset(FairseqDataset):
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
        # if lig_neighbor_seq_mask:
        #     merged_cluster_dict_f = '/home/chenty/abacust_mem/src/fasde/data/pdbs/merged_cluster_dict.npy'
        # else:
        #     merged_cluster_dict_f = '/home/chenty/abacust_mem/src/fasde/data/pdbs/merged_cluster_dict.npy'
        merged_cluster_dict_f = '/home/chenty/abacust_mem/src/data/data_storage/merged_cluster_dict.npy'

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
        self.max_lig_num = 3
        self.pick_lig_p = 0.9

        ################################### plip annotation ###################################
        # plip_ligatom_npy = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/cry_all_prot_ligatom_pair_plip_anno_update.npy'
        # self.plip_ligatom_anno_dict = np.load(plip_ligatom_npy, allow_pickle=True).item()
        # self.plip_liatom_dropout_alllig_rate = 0.5
        ################################### plip annotation ###################################

        # self.merged_cluster_center_list = self.read_cluster_center_from_txt(data_path)
        # self.data_len = len(self.merged_cluster_center_list)

        self.merged_cluster_center_list = self.read_cluster_center_from_npy(merged_cluster_dict_f, split=self.split)

        #############################################################
        # 改方案，如果是跑valid就把所有的都算一遍，之后加权
        if self.split == "train":
            self.data_len = len(self.merged_cluster_center_list)
        else:
            self.full_valid_data_name = [
                                    (pdbname,1/len(pdbname_list)) for center,pdbname_list in self.merged_cluster_dict["valid"].items()
                                    for pdbname in pdbname_list
                                    ]

            self.data_len = len(self.full_valid_data_name)
            logger.info(f"total {self.data_len} items in the valid dataset")

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
                sample_lig_num_p = (np.arange(max_lig_num)+1)/np.sum((np.arange(max_lig_num)+1))#计算概率密度，较多配体的权重更大
                cur_lig_num = (np.random.choice(np.arange(max_lig_num), 1, replace=False, p=sample_lig_num_p) + 1)[0]#返回一个长度为1的一维数组，从np.arange(max_lig_num)里随机选择，不重复
                cur_sel_lig_names = np.random.choice(lig_names, cur_lig_num, replace=False)#从lignames中随机选取cur_lig_num个进行分析，这是在给lig分析诸如随机性（真的可以吗）

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
            traceback.print_exc()
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


    def __getitem__(self, idx, strict_mode = True, crop = True):
        """fairseq会自动调用这个函数结合collator来输入样本，

        Args:
            idx (_type_): 想要获取的序列
            strict_mode (bool, optional): 如果为true则会检查输入的pdbtm张量以及pdb张量的tokens以及chains部分是否一致. Defaults to True.

        Returns:
            _type_: 单个样本字典，如果crop = true的话会对整个字典进行剪切
        """

        rep_data_name = self.merged_cluster_center_list[(idx % len(self.merged_cluster_center_list))]        #self.data_list记录着所有的pdbid名字
        weight = 0
        # rep_data_name = "6luq"
            # merged_cluster_dict = {
                #     'train':{
                #     rep_data_name_1:  [data_name_1, data_name_2, ..., data_name_n]
                #     ...
                # }
                # }
        try:
            consider_lig = True
            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
                data_name = np.random.choice(self.merged_cluster_dict[self.split][rep_data_name])
                # data_name = '2rh1'
                #从聚类结果中随机选择一个pdbid
                if self.split == 'valid':
                    logger.info("validation epoch, using valid dataset")
                    data_name, weight = self.full_valid_data_name[idx % self.data_len]
                    # logging.info(f"validating {data_name},:{weight}")
                    # weight = 1/len(self.merged_cluster_dict[self.split][rep_data_name])


            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
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
                npy_dir = Path(self.npy_path)
                pdbcode_l = pdbcode.lower()
                prot_lig_data_f = list(npy_dir.glob(f'{pdbcode_l}*.npy'))[0]
                # prot_lig_data_f = os.path.join(self.npy_path, f"{pdbcode}_complex.npy")  # 假设文件名为 {pdbcode}.npy
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
                af2_tokens = torch.tensor([restype_order[aatype_str] for aatype_str in sequence]).long()
                tokens = af2_tokens + 4
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

            ##########################################################################################################
            from protein_utils.pdbtm_data_parser import Pdbtm_parser

            pdbtm_file_dir = Path(self.pdbtm_file_path)
            pdbcode_l = pdbcode.lower()
            pdbtm_file_path = list(pdbtm_file_dir.glob(f'{pdbcode_l}*.npy'))[0]
            pdbtm_item = None
            try:
                data = np.load(pdbtm_file_path, allow_pickle=True)
                pdbtm_item = Pdbtm_parser._init_from_filepath(pdbtm_file_path)
            except:
                raise ValueError(f'{pdbtm_file_path} not exist!')

            features["tm_width"] = torch.tensor(pdbtm_item._membrane_thickness)
            features["tmatrix"] = torch.tensor(pdbtm_item._tmatrix)
            features["normal"] = torch.tensor(pdbtm_item._normal)
            features["biomatrix"] = None
            features["pdbtm_regions"] = torch.tensor(pdbtm_item.extract_full_region_np())



            # attributes = data.item().get('attributes', {})
            # membrane = attributes.get('MEMBRANE', {})

            # normal_dict = membrane.get('NORMAL', {})
            # features['normal'] = torch.tensor([float(value) for value in normal_dict.values()])
            # features['tmatrix'] = torch.tensor(membrane.get('TMATRIX', np.array([])))  # 或者 torch.empty(0)
            # # print(features["normal"],features["tmatrix"])

            # biomarix_arr =  membrane.get('BIOMATRIX', np.array([]))
            # if biomarix_arr is not None:
            #     features['biomatrix'] = torch.tensor(biomarix_arr)
            # else:
            #     features['biomatrix'] = torch.empty(0)

            #之后处理跨膜类型的信息（目前有bug）
            # tp = attributes.get('RAWRES', {})
            # if tp:
            #     features['tmtype'] = tmtype_dict.get((tp[0].get('type', 'Unknown')), -1)
            # else:
            #     features['tmtype'] = tmtype_dict.get('Unknown', -1)

            #处理regions，seq,chain信息
            # chains = data.item().get('chains', [])
            # chains.sort(key=lambda x: x['chain_id'])

            # pdbtm_tensor_list = []
            # for chain in chains:
            #     warningtype = 0
            #     try:
            #         seq_str = chain['sequence']
            #         regions = chain['regions']
            #         chain_idx = chain['chain_id']

            #         pdbtm_region_list = []

            #         for i,region in enumerate(regions):

            #             try:
            #                 if region[0] in [ 'U','0']:
            #                     continue

            #                 #拆包，获取每一条链对应的seqidx和pdbidx
            #                 seq_start, seq_end = region[1]
            #                 pdb_start, pdb_end = region[2]

            #                 # 提取子序列
            #                 sub_sequence = seq_str[seq_start-1:seq_end]#算到seq_start-1位，如果这里是1，则是从第二位开始取
            #                 sub_sequence = torch.tensor([restype_order[aatype_str] for aatype_str in sub_sequence]).long() + 4#序列
            #                 sub_idx = torch.from_numpy(np.arange(seq_start, seq_end+1))#序列idx，从1开始
            #                 pdb_sub_idx = torch.from_numpy(np.arange(pdb_start, pdb_end+1))#序列idx#pdbidx


            #                 if pdb_end>=pdb_start:
            #                     region_tensor = torch.full(((pdb_end - pdb_start+1),), region_dict[region[0]])#位置标注掩码
            #                     chain_tensor = torch.full(((pdb_end - pdb_start+1),), (get_chain_id_value(chain_idx) + 2))#位置标注掩码,a链从第二个开始\
            #                 else:
            #                     #在原始的pdb文件中，可能会有一段氨基酸在pdb中的序列突然增加一段距离之后又回来
            #                     # 例如401，102，1003.....1050，451...
            #                     #这种情况放弃以pdb作为参照，转而以seq的idx作为参照
            #                     #但最终肯定还是要以pdb的序号为准，要不然和另一部分的npy对不齐
            #                     #只是将regiontensor和chaintensor相应的量做的长度一致而已
            #                     assert seq_end-seq_start>=0,f"{pdbcode}'s {chain['chain_id']}'s region's seq invalid!"

            #                     region_tensor = torch.full(((seq_end - seq_start+1),), region_dict[region[0]])
            #                     chain_tensor = torch.full(((seq_end - seq_start+1),), (get_chain_id_value(chain_idx) + 2))

            #                 tensors_to_stack = [sub_idx, sub_sequence, pdb_sub_idx, region_tensor, chain_tensor]
            #                 lengths = [t.size(0) for t in tensors_to_stack]
            #                 #确保都是等长的张量，因为有一些蛋白seq和pdbseq是对不齐的
            #                 if len(set(lengths)) != 1:
            #                     warningtype = 2
            #                     continue

            #                 # 将子序列转换为张量
            #                 try:
            #                     result_tensor = torch.stack(tensors_to_stack, dim=1)
            #                     pdbtm_region_list.append(result_tensor)
            #                 except Exception as e:
            #                     # print([sub_idx, sub_sequence, pdb_sub_idx, region_tensor, chain_tensor])
            #                     traceback.print_exc(e)
            #                     import pdb;pdb.set_trace()


            #             except Exception as e :
            #                 traceback.print_exc(e)
            #                 import pdb;pdb.set_trace()

            #         if len(pdbtm_region_list) == 0:
            #             warningtype = 1
            #             continue
            #             # import pdb;pdb.set_trace()
            #         else:
            #             pdbtm_region_list_tensor = torch.cat(pdbtm_region_list, dim=0)
            #             pdbtm_tensor_list.append(pdbtm_region_list_tensor)
            #             #组成以列为单位的张量列表，一共有五个张量：
            #             #[sub_idx, ：从pdbtm的xml中获取的序列idx
            #             # sub_sequence, ：从pdbtm的xml中获取的序列（已经编码之后+4）
            #             # pdb_sub_idx, ：从pdbtm的xml中获取的pdb的idx，用来与pdb的data对齐
            #             # region_tensor, ：region的编码
            #             # chain_tensor ：链的标记（大概率与pdb的链一致，但有的时候会有个别增删）


            #     except Exception as e:
            #         traceback.print_exc(e)
            #         import pdb;pdb.set_trace()
            #     if warningtype == 1:
            #         logger.warning(f"{features['pdbname']}'s {chain['chain_id']} pdbtm_list is empty")
            #     elif warningtype == 2:
            #         logger.warning(f"{features['pdbname']}'s {chain['chain_id']} seq not align")


            # pdbtm_tensor_list_tensor = torch.cat(pdbtm_tensor_list,dim=0)

            #pdb_tensor是完备的，序列连续，且有gap记录，chain递增且连续
            # pdb_tensor_list = [
            #     features['chainidx'] + 2,#使得chainidx对齐
            #     features['residx'],
            #     features['tokens'],
            # ]
            # try:
            #     pdb_tensor_list = torch.stack(pdb_tensor_list,dim = 1)
            # except:
            #     import pdb;pdb.set_trace()

            # modefied_pdb_tensor, invalid_prot_mask = align_pdb_pdbtm_tensors(pdb_tensor_list,pdbtm_tensor_list_tensor,features['pdbname'])

            # import pdb;pdb.set_trace()
            # modefied_pdb_tensor = modefied_pdb_tensor.long()
            # features['residx'] = modefied_pdb_tensor[:,1]#好像还不是从1开始的吧
            # features['pdbtm_regions'] = modefied_pdb_tensor[:, 3]
            features['pdbtm_regions_len'] = len(features['pdbtm_regions'])
            #################################################################################################
            #仅保留pdbtm里面有的链
            # keep_idx = (invalid_prot_mask == 0).nonzero(as_tuple=True)[0]  # 保留的索引
            # for k, v in features.items():
            #     if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == invalid_prot_mask.shape[0]:
            #         features[k] = v.index_select(0, keep_idx)
            ###################################################################################################


            if strict_mode == True:
                if not self.check_align(features):
                    logger.warning("lig_prot_features not align!")
                    # print(features["pdbtm_regions"])
                    # print(features["residx"])
                    # print(features["tokens"])
                    print(len(features["pdbtm_regions"]))
                    print(len(features["residx"]))
                    print(pdbcode)
                    print(pdbtm_item.extract_full_seq_str(insert_code=""))
                    from protein_utils.fasta_utils import Protein_Sequence
                    residx_show = Protein_Sequence.transfer_numform_sequence_to_str(features["tokens"]-4)
                    print(residx_show)
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
                # self.max_len = 512
                features = crop_dict(features, threshold = self.max_len)#定义为256

            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
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
                # }


            lig_prot_features['pdbname'] = features['pdbname']
            lig_prot_features['normal'] = features['normal']
            lig_prot_features['tmatrix'] = features['tmatrix']
            lig_prot_features['biomatrix'] = features['biomatrix']
            lig_prot_features['pdbtm_regions'] =  torch.nn.functional.pad(features['pdbtm_regions'], (0, (lig_prot_features['coords'].shape[0]-features['pdbtm_regions'].shape[0])))#默认第二个维度是L
            lig_prot_features['pdbtm_regions_len'] = features['pdbtm_regions_len']
            lig_prot_features['edge_mem_mask'] = torch.nn.functional.pad(features['edge_mem_mask'], (0, (lig_prot_features['coords'].shape[0]-features["edge_mem_mask"].shape[0])))#默认第二个维度是L
            lig_prot_features['idx'] = torch.FloatTensor([idx])
            lig_prot_features['weight'] = weight

            return lig_prot_features

        except Exception as e:
            traceback.print_exc()
            return None


    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None
        batch = {}
        max_len = max([sample['coords'].shape[0] for sample in samples])
        #将所有类型的量从每个样本为一级目录变为每个数据类型为一级目录，
        for k in samples[0].keys():
            if k in ['tokens']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=1)
            elif k in ['lig_neighbor_seq_mask']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=True)
            elif k in ['lig_edge_index']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=-2)
            elif k in ['biomatrix']:
                pass
            elif k in ['pdbname', 'lignames','tmtype','rec_len']:
                batch[k] = [s[k] for s in samples]
            else:
                try:
                    batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=0)
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
                        print(f"{pdbname}--{key}: torch.where(tm_chain[:, 2] == current_idx)[0]: { torch.where(tm_chain[:, 2] == current_idx)[0]}")
                        traceback.print_exc(e)


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


def pad_and_stack(tensors, batch_first = True, value=0):
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




class MixedPDBAFDBDataset(FullAtomDataset):
    def __init__(
        self,
        seed,
        data_path,
        pdbtm_file_path,
        pdb_path,
        npy_path,
        split,
        config = None,
        mode = "train",
        crop_size = 256,
        resize_len = False,


        afdb_npy_path = None, afdb_list_path = None, afdb_tmdet_npy_path = None,
        afdb_plddt_threshold = 70.0, afdb_ptm_threshold = 0.5,
        pdb_ratio = 0.5,                     # PDB 采样比例 (0-1)
        afdb_noise_sigma = 0.2,              # AFDB 骨架噪声水平
        afdb_apply_noise_prob = 1,         # 对 AFDB 应用噪声的概率
        **kwargs
    ):
        super().__init__(
            seed=seed, data_path=data_path,
            pdbtm_file_path=pdbtm_file_path, pdb_path=pdb_path,
            npy_path=npy_path, split=split,
            config=config,
            mode=mode,
            crop_size=crop_size,
            resize_len=resize_len,
            **kwargs
        )

        self.pdb_ratio = pdb_ratio
        self._get_pdb_item = FullAtomDataset.__getitem__.__get__(self, MixedPDBAFDBDataset)
        self._pdb_len = FullAtomDataset.__len__.__get__(self, MixedPDBAFDBDataset)

        self.afdb_npy_path = Path(afdb_npy_path)
        self.afdb_list_path = Path(afdb_list_path)
        self.afdb_tmdet_npy_path = Path(afdb_tmdet_npy_path)

        self.afdb_plddt_threshold = afdb_plddt_threshold
        self.afdb_ptm_threshold = afdb_ptm_threshold
        self.afdb_noise_sigma = afdb_noise_sigma
        self.afdb_apply_noise_prob = afdb_apply_noise_prob

        if self.afdb_npy_path and self.afdb_npy_path.exists():
            self.afdb_merged_cluster_center_dict = self._load_afdb_index()
            self.afdb_merged_cluster_center_list = [center for center, item in self.afdb_merged_cluster_center_dict.items()]

        # 设置随机种子
        self.rng = np.random.RandomState(seed + 42)  # 不同偏移避免与父类冲突

        self.set_epoch(1)
        self.__getitem__(0)
        logger.info(f"there are {self.__len__()} item in a epoch")


    def _load_afdb_index(self):
        """加载并筛选 AFDB 数据"""
        import pickle

        data = np.load(self.afdb_list_path, allow_pickle=True).item()
        return data if isinstance(data, dict) else pickle.loads(data)


    def _add_noise_to_coords(self, coords, sigma: float):
        if sigma <= 0:
            return coords
        noise = self.rng.normal(0, sigma, coords.shape)
        return coords + noise

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_afdb_item(self, idx, crop = True):
        """加载单个 AFDB 样本"""

        rep_data_name = self.afdb_merged_cluster_center_list[(idx % len(self.afdb_merged_cluster_center_list))]
        # import pdb; pdb.set_trace()
        # rep_data_name = "6luq"
            # merged_cluster_dict = {
                #     'train':{
                #     rep_data_name_1:  [data_name_1, data_name_2, ..., data_name_n]
                #     ...
                # }
                # }
        try:
            consider_lig = True
            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
                data_name = np.random.choice(self.afdb_merged_cluster_center_dict[rep_data_name])
                # data_name = '2rh1'


            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
                # prot_lig_data_dict 示例格式：{
                #     'prot': {
                #         'atom_bb': np.ndarray,  # 主链原子坐标，形状为 (M, L, N, 3)
                #         'atom_37': np.ndarray,  # 37 个常见原子坐标，形状为 (M, L, N, 3)
                #         'pdbres_idx': np.ndarray,  # 残基索引信息
                #         'sequence': str  # 氨基酸序列
                #     },
                #     'lig': list  # 配体相关数据列表
                # }
                pdbcode = data_name[:-2]
                ##################################################################################
                # 获取npy文件
                npy_dir = Path(self.afdb_npy_path)
                prot_lig_data_f = npy_dir / f'{pdbcode}_tr.npy'

                prot_lig_data_dict = np.load(prot_lig_data_f, allow_pickle=True).item()
                ##################################################################################
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
                af2_tokens = torch.tensor([restype_order[aatype_str] for aatype_str in sequence]).long()
                tokens = af2_tokens + 4
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

            ##########################################################################################################
            # import ipdb; ipdb.set_trace()
            from protein_utils.pdbtm_data_parser import Pdbtm_parser

            pdbtm_file_dir = Path(self.afdb_tmdet_npy_path)
            afdb_tmdet_npy_path = pdbtm_file_dir / f'{pdbcode}_tmdet.npy'
            pdbtm_item = None
            try:
                # data = np.load(afdb_tmdet_npy_path, allow_pickle=True)
                pdbtm_item = Pdbtm_parser._init_from_filepath(afdb_tmdet_npy_path)
            except:
                raise ValueError(f'{afdb_tmdet_npy_path} not exist!')

            features["tm_width"] = torch.tensor(pdbtm_item._membrane_thickness)
            features["tmatrix"] = torch.tensor(pdbtm_item._tmatrix)
            features["normal"] = torch.tensor(pdbtm_item._normal)
            features["biomatrix"] = None
            features["pdbtm_regions"] = torch.tensor(pdbtm_item.extract_full_region_np())
            features['pdbtm_regions_len'] = len(features['pdbtm_regions'])




            if not self.check_align(features):
                logger.warning("lig_prot_features not align!")
                print(len(features["pdbtm_regions"]))
                print(len(features["residx"]))
                print(pdbcode)
                print(pdbtm_item.extract_full_seq_str(insert_code=""))
                from protein_utils.fasta_utils import Protein_Sequence
                residx_show = Protein_Sequence.transfer_numform_sequence_to_str(features["tokens"]-4)
                print(residx_show)
                import pdb;pdb.set_trace()

            char2weight = {k: 1 if k in {'H','B','I','C','L'} else 0 for k in region_dict.keys()}
            # 2. 把映射表转成 tensor，方便一次性索引，假设字符的整数编码就是 region_dict 的值（0~9）
            weight_table = torch.tensor(
                    [char2weight[reverse_region_dict[i]] for i in range(len(region_dict))],
                    dtype=torch.float32
                )
            tm_region_mask = weight_table[features['pdbtm_regions']]
            boundary_mask_left = (tm_region_mask.long()[2:] != tm_region_mask.long()[:-2])  # 左侧交界
            boundary_mask_right = (tm_region_mask.long()[:-2] != tm_region_mask.long()[2:])  # 右侧交界


            edge_mem_mask = torch.zeros_like(tm_region_mask).long()
            edge_mem_mask[2:] = torch.where(boundary_mask_left, torch.tensor(1, device=edge_mem_mask.device), edge_mem_mask[2:])
            edge_mem_mask[:-2] = torch.where(boundary_mask_right, torch.tensor(1, device=edge_mem_mask.device), edge_mem_mask[:-2])

            features['edge_mem_mask'] = edge_mem_mask

            if crop:
                features = crop_dict(features, threshold = self.max_len)#定义为256

            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
                if (len(prot_lig_data_dict['lig']) > 0):
                    if (np.random.rand(1)[0] < self.pick_lig_p):
                        consider_lig = True
                    else:
                        consider_lig = False
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
                # }
            if self.rng.random() <= self.afdb_apply_noise_prob: #根据预设比例加噪
                lig_prot_features["noisy"] = True
                lig_prot_features["noise_sigma"] = self.afdb_noise_sigma

            else:
                lig_prot_features["noisy"] = False
                lig_prot_features["noise_sigma"] = 0.0


            lig_prot_features['pdbname'] = features['pdbname']
            lig_prot_features['normal'] = features['normal']
            lig_prot_features['tmatrix'] = features['tmatrix']
            lig_prot_features['biomatrix'] = features['biomatrix']
            lig_prot_features['pdbtm_regions'] =  torch.nn.functional.pad(features['pdbtm_regions'], (0, (lig_prot_features['coords'].shape[0]-features['pdbtm_regions'].shape[0])))#默认第二个维度是L
            lig_prot_features['pdbtm_regions_len'] = features['pdbtm_regions_len']
            lig_prot_features['edge_mem_mask'] = torch.nn.functional.pad(features['edge_mem_mask'], (0, (lig_prot_features['coords'].shape[0]-features["edge_mem_mask"].shape[0])))#默认第二个维度是L
            lig_prot_features['idx'] = torch.FloatTensor([idx])
            lig_prot_features['weight'] = 1


            return lig_prot_features

        except Exception as e:
            traceback.print_exc()
            return None


    def __len__(self):
        if self.mode == "train":
            return len(self.afdb_merged_cluster_center_list)
        else:
            return self._pdb_len()

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
        # print(self.rng.random())

        if self.mode != "train":
            return self._get_pdb_item(index, strict_mode = True, crop = True)

        if self.rng.random() < self.pdb_ratio:
            return self._get_pdb_item(index, strict_mode = True, crop = True)
        else:
            return self._get_afdb_item(index, crop = True)


    @staticmethod
    def collater(samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None
        batch = {}
        max_len = max([sample['coords'].shape[0] for sample in samples])
        #将所有类型的量从每个样本为一级目录变为每个数据类型为一级目录，
        for k in samples[0].keys():
            if k in ['tokens']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=1)
            elif k in ['lig_neighbor_seq_mask']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=True)
            elif k in ['lig_edge_index']:
                batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=-2)
            elif k in ['biomatrix']:
                pass
            elif k in ['pdbname', 'lignames','tmtype','rec_len']:
                batch[k] = [s[k] for s in samples]
            elif k in ["weight"]:
                batch[k] = pad_and_stack([s.get(k, 1) for s in samples], batch_first=True, value=0)
            elif k in ["noisy"]:
                batch[k] = pad_and_stack([s.get(k, True) for s in samples], batch_first=True, value=0)
            elif k in ["noise_sigma"]:
                batch[k] = pad_and_stack([s.get(k, 0) for s in samples], batch_first=True, value=0)
            else:
                try:
                    batch[k] = pad_and_stack([s[k] for s in samples], batch_first=True, value=0)
                except Exception as e:
                    traceback.print_exc()
                    import pdb;pdb.set_trace()
        # import ipdb; ipdb.set_trace()
        return batch

#测试部分：

def test2():
    seed = 43
    np.random.seed(seed)
    torch.manual_seed(seed)

    pdbtm_file_path = '/home/chenty/abacust_mem/src/data/prot_lists/pdbtm_data/tm_npys/'
    pdb_path = '/home/chenty/abacust_mem/src/data/pdbs/'
    npy_path = '/home/chenty/abacust_mem/src/fasde/data/pdbs/all_npy'
    data_path = '/home/chenty/abacust_mem/src/fasde/data/pdbs/cluster_center.txt'
    split = 'train'
    tmprdataset = FullAtomDataset(
        seed,
        data_path,
        pdbtm_file_path,
        pdb_path,
        npy_path,
        split,
        )

    def test(item):
        result = tmprdataset.__getitem__(item,crop=False)
        name = result['pdbname'][0]
        print(name)
        try:
            from ...inference.pdb_generation import PDBgen
        except Exception as e :
            sys.path.append('/home/chenty/abacust_mem/src/inference')
            from pdb_generation import PDBgen


        output_pdb = f"/home/chenty/abacust_mem/src/inference/demo/{name}_output.pdb"
        coords = torch.unsqueeze(result['coords'], dim=0)
        tokens = torch.unsqueeze(result['tokens'], dim=0)
        PDBgen.tensor_to_pdb(coords, tokens, output_pdb)
        affine = result["tmatrix"]
        normal = result['normal']
        # write_dict_to_txt(result["residx"], "output.txt")
        # PDBgen.add_ag_square_grid(output_pdb, affine, normal=normal, pdb_out = output_pdb)

        # result = tmprdataset.__getitem__(item)

        # print(result["lig_edge_index"])
    for i in range(1):
        np.random.seed(i)
        torch.manual_seed(i)
        test(np.random.randint(0, 1001))




if __name__ == "__main__":
    test2()