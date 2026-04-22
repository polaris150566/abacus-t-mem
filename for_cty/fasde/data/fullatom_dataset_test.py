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
from .utils.diffusion_utils_atoms_test import make_complex_feature, make_protein_feature, make_af_feature
from .utils import features,target_features
from ..modules import gvp_modules
from ..modules.sde_lib import VPSDE, VESDE
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
path='/train14/superbrain/lili27/protein/alphafold2/savedir/bfloat16/config.json'
data_config=mlc.ConfigDict(json.loads(open(path).read()))

logger = logging.getLogger(__file__)

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
        fix_receptor_backbone = True,
        only_prot = False
    ):
        super().__init__()
        self.only_prot = only_prot
        self.data_path = data_path
        self.pdb_path = pdb_path
        self.pdbbind_path = pdbbind_path
        self.max_len = crop_size
        self.afdb_path = '/train14/superbrain/lhchen/data/alphafold_database/data'
        self.resize_len = resize_len
        self.lig_single_dim = lig_single_dim
        self.lig_pair_dim = lig_pair_dim
        self.num_aatypes = num_aatypes

        self.data_list = [k.strip().split() for k in open(f'{self.data_path}/{split}.txt')]
        self.data_len = len(self.data_list)

        # self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max)
        prot_sde = VESDE(rec_beta_min, rec_beta_max)
        lig_sde = VESDE(lig_beta_min, lig_beta_max)
        self.sde = (prot_sde, lig_sde )
        
        self.seed = seed
        self.mode = mode

        self.use_esm=False
        self.fix_receptor_backbone=fix_receptor_backbone
        self.set_epoch(1)
        if split=='valid':
            self.__getitem__(0)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def norm_coords(self, data):
        coords = data['coords']['multichain_merged_coords']
        coords = coords - coords[:, 1].mean(0)[None, None]
        data['coords']['multichain_merged_coords'] = coords
        return data

    def crop_data(self, data):
        coords = data['coords']
        esm = data['esm']
        L = len(coords['sequence'])

        crop_len = self.max_len - 2 
        if L <= crop_len:
            offset = 0
        else:
            offset = np.random.randint(0, L - crop_len)
        crop_coords = coords['multichain_merged_coords'][offset:offset+crop_len]
        new_data = {
            'coords': {
                'multichain_merged_coords': torch.FloatTensor(crop_coords),
                'multichain_length_dict': {k: crop_len for k, v in coords['multichain_length_dict'].items()},
                'sequence': coords['sequence'][offset:offset+crop_len],
                'sstype': torch.LongTensor(coords['sstype'][offset:offset+crop_len]),
                'pdbresID': {k: v[offset:offset+crop_len] for k, v in coords['pdbresID'].items()},
                'merged_chain_label': coords['merged_chain_label'][offset:offset+crop_len],
            },
            'esm': {
                'esm_assembly_rep': torch.FloatTensor(esm['esm_assembly_rep'][offset:offset+crop_len]),
                'esm_assembly_single_mask': torch.FloatTensor(esm['esm_assembly_single_mask'][offset:offset+crop_len]),
            }
        }
        return new_data

    def resize(self, mat):
        if not self.resize_len:
            return mat
        
        L = len(mat)
        if L % self.downsample_scale == 0:
            return mat

        if isinstance(mat, str):
            return mat + ''.join(['X'] * (L - len(mat)))
        
        L = (L // self.downsample_scale + 1) * self.downsample_scale
        return data_utils.pad_to_length(mat, L, 0, 0)

    def get_pdb_data(self, pdbcode):
        data_path = f'{self.pdb_path}/{pdbcode[1:3]}/{pdbcode}.npy'
        raw_data = np.load(data_path, allow_pickle=True).item()
        raw_data = self.crop_data(raw_data)

        esm = self.resize(raw_data['esm']['esm_assembly_rep'])
        # Prior data
       
        chain_id = [c for c in raw_data['coords']['pdbresID']][0]
        atom14 = raw_data['coords']['multichain_merged_coords']
        sequence = raw_data['coords']['sequence']
        mask14 = (atom14 != 0).any(-1)
        residx = np.array(raw_data['coords']['pdbresID'][chain_id], dtype=np.int32)

        targets={'all_atom_positions':atom14,'all_atom_mask':mask14,'sequence':sequence}
        raw_features = {**targets,**features.make_sequence_features(sequence=sequence, description='A', num_res=len(sequence)),}
        raw_features = {k: to_tensor(v) for k, v in raw_features.items()}
        feature_dict = features.process_features(raw_features, data_config)
        num_rec = feature_dict.pop('num_recycle')
        feature_dict = {k:v[0] for k,v in feature_dict.items()}
        atom37 = all_atom.atom14_to_atom37(feature_dict['all_atom_positions'],feature_dict)
        feature_dict['all_atom_positions'] = atom37
        feature_dict['all_atom_mask'] = feature_dict['atom37_atom_exists']
        feature_dict = {k:v[None]  for k,v in feature_dict.items()}
        feature_dict['num_recycle'] = 1
        rec_feature = target_features.make_target_features(feature_dict, 2.638, use_clamped_fape=True)
        rec_feature.pop('num_recycle')
        for k,v in rec_feature.items():
            if len(v.shape)>1:
                rec_feature[k]=v[0]
            else:
                rec_feature[k]=v
        feature_dict = {
            "esm": esm,
            'residue_index': torch.from_numpy(residx),
            'residx': torch.from_numpy(residx),
            'chainidx': torch.tensor(raw_data['coords']['merged_chain_label']),
            **rec_feature,
            'has_ligand': torch.Tensor([False])
        }
        return feature_dict, None

    def crop_receptor(self, ca_coord, lig_pos):
        ca_coord = torch.FloatTensor(ca_coord)
        lig_len = lig_pos.shape[0]
        crop_len = self.max_len - lig_len
        rec_len = ca_coord.shape[0]
        if rec_len <= crop_len:
            return np.arange(rec_len)

        dist = torch.sqrt(torch.sum((ca_coord[:, None] - lig_pos[None])**2, -1)).min(-1)[0]
        sort_index = torch.argsort(dist)[:crop_len]
        crop_index = torch.sort(sort_index)[0]
        return crop_index.tolist()
        
        
    def get_pdbbind_data(self, pdbcode):
        raw_data = np.load(f'{self.pdbbind_path}/{pdbcode}.npy', allow_pickle=True).item()

        atom14 = raw_data['complex']['rec_fullatom']['atom14'].astype(np.float32)
        crop_index = self.crop_receptor(
            atom14[:, 1], raw_data['complex']['ligand'].pos
        )
        atom14 = atom14[crop_index]
        rec_esm = raw_data['complex']['receptor']['x'][crop_index, 1:]
        mask14 = raw_data['complex']['rec_fullatom']['atom14_mask'][crop_index].astype(np.float32)
        sequence = raw_data['complex']['rec_fullatom']['sequence']
        sequence = ''.join([sequence[idx] for idx in crop_index])
        residx = raw_data['complex']['rec_fullatom']['residx'][crop_index].astype(np.int32)
        chainidx = raw_data['complex']['rec_fullatom']['chainidx'][crop_index].astype(np.int32)
        # import pdb; pdb.set_trace(
        targets={'all_atom_positions':atom14,'all_atom_mask':mask14,'sequence':sequence}
        raw_features = {**targets,**features.make_sequence_features(sequence=sequence, description='A', num_res=len(sequence)),}
        raw_features = {k: to_tensor(v) for k, v in raw_features.items()}
        feature_dict = features.process_features(raw_features, data_config)
        num_rec = feature_dict.pop('num_recycle')
        feature_dict = {k:v[0] for k,v in feature_dict.items()}
        atom37 = all_atom.atom14_to_atom37(feature_dict['all_atom_positions'],feature_dict)
        feature_dict['all_atom_positions'] = atom37
        feature_dict['all_atom_mask'] = feature_dict['atom37_atom_exists']
        feature_dict = {k:v[None]  for k,v in feature_dict.items()}
        feature_dict['num_recycle'] = 1
        rec_feature = target_features.make_target_features(feature_dict, 2.638, use_clamped_fape=True)
        rec_feature.pop('num_recycle')
        for k,v in rec_feature.items():
            if len(v.shape)>1:
                rec_feature[k]=v[0]
            else:
                rec_feature[k]=v
        rec_feature.update({
            'residx': torch.from_numpy(residx),
            'chainidx': torch.LongTensor(chainidx),
            'esm': rec_esm,
            'has_ligand': torch.Tensor([True]),
        })

        return rec_feature, raw_data['complex']

    def __getitem__(self, idx):
        
        try:
        # if True:
            with data_utils.numpy_seed(self.seed, self.epoch * self.data_len + idx):
                if len(self.data_list[idx])==2:
                    data_name, data_type = self.data_list[idx]
                    if data_type == 'P':
                        rec, raw_data = self.get_pdb_data(data_name)
                        features = make_protein_feature(
                        rec, raw_data,
                        self.lig_single_dim, self.lig_pair_dim,
                        self.sde,
                        num_aatypes = self.num_aatypes,
                        fix_receptor_backbone = self.fix_receptor_backbone,
                        use_esm=self.use_esm
                    )
                    elif data_type == 'C':

                        rec, raw_data = self.get_pdbbind_data(data_name)
                        if self.only_prot:
                            features = make_protein_feature(
                                rec, raw_data,
                                self.lig_single_dim, self.lig_pair_dim,
                                self.sde,
                                num_aatypes = self.num_aatypes,
                                fix_receptor_backbone = self.fix_receptor_backbone,
                                use_esm=self.use_esm
                                )
                        else:
                            features = make_complex_feature(
                            rec, raw_data,
                            self.lig_single_dim, self.lig_pair_dim,
                            self.sde,
                            num_aatypes = self.num_aatypes,
                            fix_receptor_backbone = self.fix_receptor_backbone,
                            use_esm=self.use_esm
                            )
                    else:
                        raise NotImplementedError
                else:
                    data_list_split = self.data_list[idx]
                    uniprot_id, tax_id, *_ = data_list_split
                    data_name = uniprot_id
                    tax_id_prefix = tax_id.split('-')[0]
                    cif_path = f'{self.afdb_path}/{tax_id_prefix[:3]}/{tax_id}/AF-{uniprot_id}-F1-model_v3.cif.gz'
                    coords, sequence, residx, confidence = gvp_modules.util.load_coord_from_cif_zip(cif_path, 'A')
                 
                    if coords.shape[0] > self.max_len:
                        crop_start_index = np.random.choice(coords.shape[0] - self.max_len)
                        coords = coords[crop_start_index:crop_start_index+self.max_len]
                        sequence = sequence[crop_start_index:crop_start_index+self.max_len]
                        confidence = confidence[crop_start_index:crop_start_index+self.max_len]
                        residx = residx[crop_start_index:crop_start_index+self.max_len]
                    
                    # aatype = residue_constants.sequence_to_onehot(sequence=sequence, mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True).numpy()
                    aatype = torch.tensor([residue_constants.restypes.index(i) for i in sequence])
                    sample = {
                        'coords': torch.from_numpy(coords),
                        'sequence': sequence,
                        'aatype': aatype,
                        'confidence':torch.from_numpy(confidence),
                        'residx': torch.from_numpy(residx),
                    }
                    features = make_af_feature(
                        sample,
                        num_aatypes = self.num_aatypes,
                        fix_receptor_backbone = self.fix_receptor_backbone,
                        use_esm=self.use_esm)
            features['idx'] = torch.FloatTensor([idx])
            features['name'] =  data_name
            return features
        except Exception:
            print('failed file : ', idx)
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
                batch[k] = data_utils.pad_and_stack([s[k] for s in samples], dim=0, value=33)
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
    elif arr.dtype == np.bool:
        return torch.BoolTensor(arr)
    else:
        return arr
