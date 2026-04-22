import json
import ml_collections as mlc
import numpy as np
import logging
import torch
import torch.nn.functional as F
from typing import *
import sys
sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
from alphafold import all_atom
from alphafold.common import residue_constants


# from .utils.protein_mpnn_utils import convert_atom14_to_atom37, sequence_to_aatype
# from .utils.permuta_ss import permute_betwee_ss_from_pos, compute_chain_center_mass, add_pseudo_c_beta_from_gly
from .utils.diffusion_utils_atoms import make_ligand_node_feature, make_complex_feature, make_protein_feature, make_af_feature, restype_order
from .utils import features,target_features
from ..modules import gvp_modules
from ..modules.sde_lib import VPSDE, VESDE
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
path='/train14/superbrain/lili27/protein/alphafold2/savedir/bfloat16/config.json'
data_config=mlc.ConfigDict(json.loads(open(path).read()))

UNIMOL_REPRS_DIM=512

logger = logging.getLogger(__file__)



plip_ligatom_anno = {
    'mask': 0, 'unknown': 1, 'pistacking': 2, 'pication_laro': 3, 'pication_paro': 4, 'hydrophobic': 5, 'hbond_ldon': 6, 'hbond_pdon': 7, 'metalcomplex': 8, 'saltbridge_lneg': 9, 'saltbridge_pneg': 10, 'halogen': 11, 'waterbridge': 12
}
plip_liatom_dropout_ligatom_rate = {'pistacking': 0.4, 'pication_laro': 0.4, 'pication_paro': 0.4, 'hydrophobic': 0.5, 'hbond_ldon': 0.2, 'hbond_pdon': 0.2, 'metalcomplex': 0.0, 'saltbridge_lneg': 0.2, 'saltbridge_pneg': 0.2, 'halogen': 0.0, 'waterbridge': 1.0}



class FullAtomDataset(FairseqDataset):
    def __init__(
        self,
        seed,
        data_path,
        pdb_path,
        pdbbind_path,
        split,
        config,
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
        config = config.data
        self.config = config
        self.data_path = data_path
        self.pdb_path = pdb_path
        self.pdbbind_path = pdbbind_path
        self.crossdock_path = '/train14/superbrain/lili27/protein/data/crossdock_processing'
        # self.pdbdock_path = '/train14/superbrain/lili27/data/prot_with_ligands_v2'
        self.pdbdock_path = '/raw22/superbrain/permanent/yfliu25/dataset/all_pdb_raw_lig'
        self.max_len = crop_size

        self.trans_scale_factor = config.globals.trans_scale_factor
        self.use_clamped_fape_prob = config.globals.use_clamped_fape_prob
        self.downsample_scale = config.globals.downsample_scale
        self.resize_len = resize_len
        self.freeze_receptor = config.globals.freeze_receptor
        self.ca_radius = ca_radius
        self.lig_center_mask_p = lig_center_mask_p
        prot_cluster_dict_f = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/all_pdb_cluster_0.3.npy'
        complex_cluster_dict_f = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/complex_pdb_cluster_0.3.npy'
        if lig_neighbor_seq_mask:
            merged_cluster_dict_f = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/all_complex_cluster_0.3.npy'
        else:
            merged_cluster_dict_f = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/all_1pdb_3complex_cluster_0.3.npy'
        self.prot_cluster_dict = np.load(prot_cluster_dict_f, allow_pickle=True).item()
        self.complex_cluster_dict = np.load(complex_cluster_dict_f, allow_pickle=True).item()
        self.merged_cluster_dict = np.load(merged_cluster_dict_f, allow_pickle=True).item()
        self.lig_single_dim = lig_single_dim
        self.lig_pair_dim = lig_pair_dim
        self.num_aatypes = num_aatypes
        self.embed_unimol_reprs = embed_unimol_reprs
        self.max_lig_num = 3
        self.pick_lig_p = 0.9

        ################################### plip annotation ###################################
        plip_ligatom_npy = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/data_list/pdb_complex_merge/utils/cry_all_prot_ligatom_pair_plip_anno_update.npy'
        self.plip_ligatom_anno_dict = np.load(plip_ligatom_npy, allow_pickle=True).item()
        self.plip_liatom_dropout_alllig_rate = 0.5
        ################################### plip annotation ###################################

        self.data_list = [k.strip().split() for k in open(f'{self.data_path}/{split}.txt')]
        self.data_len = len(self.data_list)
        
        self.seed = seed
        self.mode = mode

        self.set_epoch(1)
        if split=='train':
            self.__getitem__(0)

    def set_epoch(self, epoch):
        self.epoch = epoch

    
    def get_lig_ctx(self, lig_dict, features, consider_lig=True, max_lig_num=3):
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
                sample_lig_num_p = (np.arange(max_lig_num)+1)/np.sum((np.arange(max_lig_num)+1))
                cur_lig_num = (np.random.choice(np.arange(max_lig_num), 1, replace=False, p=sample_lig_num_p) + 1)[0]
                cur_sel_lig_names = np.random.choice(lig_names, cur_lig_num, replace=False)
                
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
                    lig_edge_index = lig_edge.edge_index + lig_atom_num
                    lig_edge_attr = lig_edge.edge_attr

                    lig_single_list.append(lig_single)
                    lig_single_unimol_list.append(lig_unimol_single)
                    lig_edge_index_list.append(lig_edge_index)
                    lig_edge_attr_list.append(lig_edge_attr)
                    lig_coords_list.append(raw_data['ligand']['pos'])

                    ################################### plip annotation ###################################
                    cur_lig_anno_itype = torch.ones(cur_lig_atoms_num)
                    query_lig_name = '_'.join(lig_name.split('.')[0].split('_')[:-1])
                    if self.plip_ligatom_anno_dict.__contains__(features['pdbname'][0] + '_1_cry'):
                        cur_pdb_ligatom_plip_anno = self.plip_ligatom_anno_dict[features['pdbname'][0] + '_1_cry']
                        if cur_pdb_ligatom_plip_anno.__contains__(query_lig_name):
                            cur_lig_atom_plip_anno_dict = cur_pdb_ligatom_plip_anno[query_lig_name]
                            if (np.random.rand(1)[0] > self.plip_liatom_dropout_alllig_rate):
                                for plip_type, pres_latom_dict in cur_lig_atom_plip_anno_dict.items():
                                    plip_type_embed_id = plip_ligatom_anno[plip_type]
                                    if (np.random.rand(1)[0] < plip_liatom_dropout_ligatom_rate[plip_type]):
                                        continue
                                    for chainres, atom_raw_id_lst in pres_latom_dict.items():
                                        for atom_raw_id in atom_raw_id_lst:
                                            cur_lig_anno_itype[atom_raw_id-1] = plip_type_embed_id
                    lig_plip_anno_itype_list.append(cur_lig_anno_itype)
                    ################################### plip annotation ###################################
                    lig_atom_num += cur_lig_atoms_num

                ################################### plip annotation ###################################
                if len(lig_plip_anno_itype_list) > 0:
                    plip_anno_itype_list = torch.cat([torch.zeros(prot_len), torch.cat(lig_plip_anno_itype_list)], dim=0)
                else:
                    plip_anno_itype_list = torch.zeros(prot_len)
                ################################### plip annotation ###################################

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

                residx = torch.cat([prot_residx, torch.arange(lig_atom_num ) + prot_residx.max() + 100], dim=0)
                residx_=(residx - torch.min(residx) + 1).tolist()
                residx=torch.tensor(residx_)
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
        lig_prot_features['plip_anno_itype_list'] = plip_anno_itype_list

        lig_prot_features['chi_angles'] = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
        lig_prot_features['alt_chi_angles'] = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
        lig_prot_features['chi_mask'] = chi_mask
        lig_prot_features['backbone_affine_tensor'] = backbone_affine_tensor
        lig_prot_features['backbone_angles_sin_cos'] = backbone_angles_sin_cos
        lig_prot_features['coords37'] = coords37
        
        return lig_prot_features


    
    def __getitem__(self, idx):
        rep_data_name = self.data_list[idx][0]
        try:
            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):   
                data_name = np.random.choice(self.merged_cluster_dict[rep_data_name]['rep'])
            if (len(data_name.split('_')) > 2):
                consider_lig = True
            else:
                consider_lig = False
            
            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):   
                pdbcode, chain = data_name.split('_')[:2]
                prot_lig_data_f = f'{self.pdb_path}/{pdbcode[1:3]}/{pdbcode}_{chain}_1.npy'
                prot_lig_data_dict = np.load(prot_lig_data_f, allow_pickle=True).item()

                prot_data_dict = prot_lig_data_dict['prot']
                atom_bb = prot_data_dict['atom_bb']
                atom_37 = prot_data_dict['atom_37']
                pdbresidx = prot_data_dict['pdbres_idx'][chain]
                sequence = prot_data_dict['sequence']
                prot_len = len(sequence)

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

                features['coords'] = atom_bb[:, [0, 1, 2, 4]]
                features['coords37'] = atom_37
                features['tokens'] = tokens
                features['node_mask'] = node_mask
                features['residx'] = pdbresidx
                features['chainidx'] = node_mask.long()
                features['pdbname'] = ['_'.join([pdbcode, chain])]
                features.update(frame_results)
                if (len(prot_lig_data_dict['lig']) > 0):
                    if (np.random.rand(1)[0] < self.pick_lig_p):
                        consider_lig = True
                lig_prot_features = self.get_lig_ctx(prot_lig_data_dict['lig'], features, consider_lig=consider_lig, max_lig_num=self.max_lig_num)

            lig_prot_features['idx'] = torch.FloatTensor([idx])
            return lig_prot_features

        except Exception:
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
        for k in samples[0].keys():
            if k in ['tokens']:
                batch[k] = data_utils.pad_and_stack([s[k] for s in samples], dim=0, value=1)
            elif k in ['lig_neighbor_seq_mask']:
                batch[k] = data_utils.pad_and_stack([s[k] for s in samples], dim=0, value=True)
            elif k in ['lig_edge_index']:
                batch[k] = data_utils.pad_and_stack([s[k] for s in samples], dim=0, value=-2)
            elif k in ['pdbname', 'lignames']:
                batch[k] = [s[k] for s in samples]
            elif k in ['rec_len']:
                batch[k] = [s[k] for s in samples]
            else:
                try:
                    batch[k] = data_utils.pad_and_stack([s[k] for s in samples], dim=0, value=0)
                except:
                    import pdb;pdb.set_trace()
        return batch
    
    def num_tokens(self, index):
        # return min(index, self.max_len)
        return self.max_len

    def size(self, index):
        # self.n_tokens = [min(int(s.split('\t')[-1]), self.max_len) for s in self.data_list]
        # self.num_samples = len(self.data_list)
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
