################################################################################################
# 根据pdb文件生成相应的npy文件输入给abacust进行训练/推理
# 分为train和inference模式，train用于对一个目录下大量pdb文件进行批量生成npy，inference则用于处理输入目录下的少量样本
# 输出的时候会在output_dir下面建立all_lig和 all_npy文件夹，all_npy目录下是{pdbname}.npy
################################################################################################
import os, sys, traceback, builtins
import warnings
import argparse
from tqdm import tqdm
import copy
from pathlib import Path
from joblib import Parallel, delayed
import glob
import numpy as np
import subprocess
import torch
import glob, os, itertools
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
import logging
from Bio.PDB import PDBParser
import builtins
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# sys.path.append("/home/chenty/abacust_mem/src/utils")
sys.path.append("/home/chenty/abacust_mem/src/data/data_utils/utils")

from unimol.extract_rep import get_unimol_repr_from_pos

from protein_coord_parser import ProteinCoordsParser

from select_hetatm import process

cand_lig_f_type = ['sdf', 'mol2', 'pdb', 'pdbqt']
ignore_mol_type = ["HOH", "NA", "CL", "K", "BR"]


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def convert_dative_bonds(bond):
    """用于将某些类型的化学键转换为明确的键类型（如单键、双键等）。
    该函数的目的是处理那些可能在 RDKit 中未明确指定的键类型（如三键或其他特殊键），并根据键的属性和原子的特性来确定更合适的键类型。

    Args:
        bond (_type_):一个表示键的对象（bond）

    Returns:
        _type_: 返回键的类型（BT）
    """
    if bond.GetBondTypeAsDouble() == 1.0 and bond.GetBeginAtom().GetTotalValence() == 1:
        return BT.SINGLE#表示单键类型
    elif bond.GetBondTypeAsDouble() == 1.0 and (bond.GetBeginAtom().GetTotalValence() %2 == 0):
        return BT.SINGLE
    elif bond.GetBondTypeAsDouble() == 1.0 and (bond.GetBeginAtom().GetTotalValence() %2 == 1):
        return BT.DOUBLE
    elif bond.GetBondTypeAsDouble() == 2.0 and bond.GetBeginAtom().GetTotalValence() == 2:
        return BT.DOUBLE
    elif bond.GetBondTypeAsDouble() == 3.0 and bond.GetBeginAtom().GetTotalValence() == 3:
        return BT.TRIPLE
    elif bond.GetBondTypeAsDouble() == 1.0 and bond.IsInRing() and bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
        return BT.AROMATIC


lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


def lig_atom_featurizer(mol):
    """解析一个mol对象，重点提取其ring部分的结构，将其整理成一个列表返回

    Args:
        mol (_type_): mol分子对象

    Returns:
        _type_: 返回一个list。里面装着的每一个元素有明确的物理意义
    """
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            # allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_chirality_list'], atom.GetChiralTag()),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)


def get_lig_graph(mol):
    """根据输入的mol对象生成对应的lig对象
    Args:
        mol (_type_): 输入的是一个分子对象mol

    Returns:
        _type_: 返回的是一个字典：
        dict:{
            ligand:{
                atom_feats:
                lig_coords:
            }
            (ligand,lig_bond,ligand):{
                edge_inex:
                edge_attr:
            }
        }
    """
    complex_graph = HeteroData()
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    #选择一个mol对象的第一个构象的位置信息，（numpy）之后转换成浮点型张量
    atom_feats = lig_atom_featurizer(mol)
    #获取atomfeats，是一个list

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        #对一个分子的所有化学键进行遍历
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #获取起始和结束的节点，装进数组中
        row += [start, end]
        col += [end, start]
        try:
            edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]#BT.UNSPECIFIED表示未知的键，可能是键类型未知或者文件损坏
        except:
            edge_type += 2 * [bonds[convert_dative_bonds(bond)]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]
        #处理键的特征，标注是单键双键还是其他类型
    edge_index = torch.tensor([row, col], dtype=torch.long)
    #将边的起始和终止组织成一个（E,2）的张量
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    #将边的特征（键数）组织成一个张量
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    #对edgetype进行onehot编码

    complex_graph['ligand'].x = atom_feats#原子的特征：list
    complex_graph['ligand'].pos = lig_coords#张量，mol对象的第一个构象的位置信息
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index#键的起始和终止，tensor
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr#键的键数，tensor
    return complex_graph


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """读取molecule_file路径下的分子文件并转换成一个mol分子对象返回

    Args:
        molecule_file (_type_): 输入的分子文件的路径
        sanitize (bool, optional): _是否进行结构修正_. Defaults to False.
        calc_charges (bool, optional): _是否计算电荷_. Defaults to False.
        remove_hs (bool, optional): _是否舍弃氢原子_. Defaults to False.

    Returns:
        _type_: 返回一个mol对象，包含分子信息
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:#根据实际情况的需要选择是否对分子结构进行修正，是否检查gasteiger电荷分布。以及是否舍弃氢原子
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def get_npy_from_cif(proteinfile, chain):
    """根据给定的pdb文件路径，将pdb文件解析成对应的原子位置信息，包括全原子（atom37）主链原子，等，作为一个字典返回。

    Args:
        proteinfile (_type_): _需要解析的pdb文件的路径（直接用pdbparser）
        chain (_type_): 需要解析的链

    Returns:
        _type_: 一个字典，包括atombb,atom37,seq,pabres_idx
    """
    try:
        logging.info(f"filename: {proteinfile}")
        PDBparser = ProteinCoordsParser(proteinfile, chain=chain, omit_mainatoms_missing=False)
    except Exception as e :
        import traceback;traceback.print_exc()
        return None

    atom_bb = PDBparser.chain_main_crd_array.reshape(-1,5,3)
    try:
        atom_37 = PDBparser.chain_crd_array.reshape(-1,37,3)
        sequence = PDBparser.get_sequence()
        abspdbindex,pdbidx,pdbhainmask = PDBparser.get_pdbresID2absID(chain)
        # import pdb;pdb.set_trace()
        # pdbresID = None
        # pdbresID = {
        #     chain: list(PDBparser.get_pdbresID2absID(chain).keys())
        #     }
        # print(f'pdbresid:{pdbresID}')
    except Exception as e:
        print (e)

    data_dict = {
        'atom_bb': atom_bb,
        'atom_bb_len':len(atom_bb),
        'atom_37': atom_37,
        'sequence': sequence,
        'abs_pdb_index':abspdbindex,
        'pdbres_idx': pdbidx,
        'pdbres_idx_len': len(pdbidx),
        'pdb_chain_mask':pdbhainmask,
    }

    return data_dict

def cand_lig_select(coords37, ligcoords, min_dist=5.0):
    """
    如果配体的所有原子与蛋白质的所有原子的距离都大于或等于 min_dist（默认为5.0），则该配体会被忽略。
    具体来说，就是当 cand_lig_select 函数返回 False 时，配体不会被进一步处理或存储。
    """
    coords37 = torch.from_numpy(coords37).float()
    ligcoords = torch.from_numpy(ligcoords).float()
    atom14_gt_exists = (~torch.all(coords37 ==0,-1)).float()
    all_atom_dist = torch.sqrt(torch.sum((coords37[:,:,None]- ligcoords[None, None])**2,-1)) # P, 37, x
    all_atom_dist_mask = ((1 - atom14_gt_exists) * 1e6)[..., None]
    all_atom_dist = all_atom_dist + all_atom_dist_mask
    lig_neighbor_seq_mask_pocket = torch.any(torch.any(all_atom_dist < min_dist, dim=-1),dim=-1)

    return torch.any(lig_neighbor_seq_mask_pocket)



def get_args():
    parser = argparse.ArgumentParser(description='GVP Transformer Inverse Folding')
    parser.add_argument('--pdb_dir', type=str)
    parser.add_argument('--pdb_list', type=str, default=None, help='txt file with one pdb path per line, used in inference mode')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument("--protein_chain", type=str, default='A')
    parser.add_argument("--atomized_chain", type=str, default='')
    parser.add_argument("--merge_atomized_chain", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default='inference')
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--skip_existing", action='store_true', default=False)
    parser.add_argument("--debug", action ='store_true', default=False)


    args = parser.parse_args()
    return args



def make_prot_lig_complex_from_pdb(args):
    """
    根据args中的配置对，pdb目录中的所有pdb文件进行遍历，创建蛋白质主体部分和配体的文件夹，并根据每个pdb文件的名字加_complex进行命名，
    在之后将pdb文件的原子位置信息提取出来作为字典（protdata),合并到prot_lig_complex_dict 字典中
    """
#     args = {
#     "pdb_dir": "...",          # 输入 PDB 文件目录
#     "out_dir": "...",          # 输出根目录
#     "protein_chain": "...",    # 蛋白质链标识符
#     "atomized_chain": "...",   # 原子化链标识符（可选）
#     "merge_atomized_chain": ...  # 是否合并原子化链（布尔值）
# }

    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 判断文件存在时别再 listdir，直接 Path.exists
    # if (out_dir / args["pdb_f"]).exists():
    # if args["pdb_f"] in os.listdir(args["out_dir"]):
    #     return None
    try:
        pdb_dir = args["pdb_dir"]

        outdir = args["out_dir"]
        #protein_chain = args.protein_chain
        protein_chain = None
        atomized_chain = args["atomized_chain"]
        index = args["index"]
        pdb_f = args["pdb_f"]
        # print(pdb_f,flush = True)
        if (len(atomized_chain) == 0):
            atomized_chain = None



        # L = len(os.listdir(pdb_dir))
        # print(f"{index}/{L}...")
        # prot_lig_pdb_f = f'{pdb_dir}/{pdb_f}'
        prot_lig_pdb_f = pdb_f   #获取pdb文件全路径
        prot_lig_pdbname = os.path.basename(prot_lig_pdb_f).split('.')[0]
        npy_f = f'{outdir}/all_npy/{prot_lig_pdbname}.npy'
        os.makedirs(f'{outdir}/all_npy', exist_ok=True)
        lig_dir = f'{outdir}/all_lig/{prot_lig_pdbname}'
        os.makedirs(lig_dir, exist_ok=True)

        if args.get("skip_existing") and os.path.exists(npy_f):
            print(f'{prot_lig_pdb_f} skipped_existing')
            return None

        process(prot_lig_pdb_f, lig_dir, atomized_chain, args["merge_atomized_chain"])
        #输出配体文件

        lig_name_list = list(set([lig_name_f.split('.')[0] for lig_name_f in os.listdir(lig_dir)]))

        prot_data = get_npy_from_cif(prot_lig_pdb_f, protein_chain)
        if (prot_data is None):
            #print(f'{prot_lig_pdb_f} {protein_chain} load failed')
            raise Exception(f'{prot_lig_pdb_f} {protein_chain} load failed')

        prot_lig_complex_dict = {}

        prot_lig_complex_dict['prot'] = prot_data
        prot_lig_complex_dict['lig'] = {}

        for lig_name in lig_name_list:
            lig_pdbcode = lig_name.split('_')[2]
            if lig_pdbcode in ignore_mol_type:
                continue

            mol = None
            for lig_data_mode in cand_lig_f_type:#[ 'sdf', 'mol2', 'pdb','pdbqt']
                absl_lig_f = f'{lig_dir}/{lig_name}.{lig_data_mode}'

                mol = read_molecule(absl_lig_f, remove_hs=True, sanitize=True)
                if (mol is not None):
                    break

            if (mol is None):
                print(lig_name, 'unloaded')
                continue

            mol = read_molecule(absl_lig_f, remove_hs=True, sanitize=True)
            lig_coords = mol.GetConformer().GetPositions()     #获取mol分子对象中的第一个构象的原子坐标（numpy数组）
            prot_coords37 = prot_data['atom_37']
            lig_flag = cand_lig_select(prot_coords37, lig_coords, min_dist=5.0)   #ligflag标注了这个lig距离指定的蛋白质主题结构的最小距离有没有达到min_dist

            if lig_flag:
                cur_lig_graph = get_lig_graph(mol)
                #获取配体的图结构字典lig
                unimol_reprs_dict = get_unimol_repr_from_pos(cur_lig_graph['ligand'], mol)

                if (cur_lig_graph['ligand'].pos.shape[0] == unimol_reprs_dict['atomic_reprs'][0].shape[0]):
                    prot_lig_complex_dict['lig'][f'{lig_name}.{lig_data_mode}'] = {'data': cur_lig_graph, 'rdkit_ligand': mol, 'unimol_reprs': unimol_reprs_dict}
                print(f'{lig_name}.{lig_data_mode} loaded')
            else:
                print(f'{lig_name}.{lig_data_mode} ignored')

        np.save(npy_f, prot_lig_complex_dict)
        print(f'{prot_lig_pdb_f} preprocessed')
    except Exception as e:
        print(e)
        return pdb_f


if __name__ == '__main__':
    args = get_args()
    pdb_dir = args.pdb_dir

    assert args.mode is not None, "args.mode is {args.mode}"
    # args.mode = 'train'

    if args.mode == 'train':
        logging.info(f"mode : train")
        args_list = []
        # for index,pdb_f in enumerate(os.listdir(pdb_dir)):
        for index, pdb_f in enumerate(glob.glob(os.path.join(pdb_dir, '*.pdb')) +glob.glob(os.path.join(pdb_dir, '*.ent'))):
            c_arg = {**vars(args), "index": index, "pdb_f": pdb_f}# 将args转换为普通字典之后加入index和pdb_f值，处理指定的文件
            if args.debug:
                import pdb;pdb.set_trace()
            args_list.append(c_arg)

        results = Parallel(n_jobs=args.num_workers)(delayed(make_prot_lig_complex_from_pdb)(arg) for arg in tqdm(args_list, total=len(args_list)))
        # print(results)
    elif args.mode == 'inference':
        args_list = []
        if args.pdb_list:
            with open(args.pdb_list) as f:
                pdb_files = [line.strip() for line in f if line.strip()]
        else:
            pdb_files = glob.glob(os.path.join(pdb_dir, '*.pdb'))
        for index, pdb_f in enumerate(pdb_files):
            c_arg = {**vars(args), "index": index, "pdb_f": pdb_f}
            if args.debug:
                import pdb;pdb.set_trace()
            args_list.append(c_arg)
        if args.num_workers == 1:
            for arg in tqdm(args_list, total=len(args_list)):
                make_prot_lig_complex_from_pdb(arg)
        else:
            Parallel(n_jobs=args.num_workers)(delayed(make_prot_lig_complex_from_pdb)(arg) for arg in tqdm(args_list, total=len(args_list)))
    else:
        raise ValueError(f"args.mode must among train/inference but got {args.mode}")