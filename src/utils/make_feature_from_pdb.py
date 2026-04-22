import sys, os
import warnings
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs

from Bio.PDB import PDBParser

from unimol.extract_rep import get_unimol_repr_from_pos
from select_hetatm import process
sys.path.append("../../utils/")
sys.path.append("/home/chenty/ABACUST-main/utils/")

from protein_coord_parser import ProteinCoordsParser



cand_lig_f_type = ['sdf', 'mol2', 'pdb', 'pdbqt']
ligMPNN_ignore_mol_type = ["HOH", "NA", "CL", "K", "BR"]
ignore_mol_type = [
    "NUC", "ZN", "CA", "MG", "III", "MN", "FE", "CU", "SF4", "FE2", "CO", "FES", "GOL", "NA",
    "CL", "K", "CU1", "XE", "NO2", "EDO", "NI", "BR", "CD", "O", "CS", "NO", "TL", "HG", "UNL", "KR",
    "SR", "RB", "F", "AG", "AR", "AU", "MO", "SE", "GD", "YB", "VX", "SM", "LI", "RE", "N", "W", "OS",
    "HO", "PI", "EDO", "PG4", "OGA", "SO4", "HEZ", "FEO", "CL", "DMS", "ACT", "MPD", "NH2", "CUA", "SIW",
    "PGW", "IOD", "3NI", "ZRW", "78M", "UNX", "MES", "CCN", "HOH"]



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
    if bond.GetBondTypeAsDouble() == 1.0 and bond.GetBeginAtom().GetTotalValence() == 1:
        return BT.SINGLE
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
    complex_graph = HeteroData()
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        try:
            edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]
        except:
            edge_type += 2 * [bonds[convert_dative_bonds(bond)]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand'].pos = lig_coords
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return complex_graph


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
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

    try:
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


def get_npy_from_cif(poteinfile, chain):
    # try:
    PDBparser = ProteinCoordsParser(poteinfile, chain=chain, omit_mainatoms_missing=False)
    # except:
    #     return None

    atom_bb = PDBparser.chain_main_crd_array.reshape(-1,5,3)
    try:
        atom_37 = PDBparser.chain_crd_array.reshape(-1,37,3)
        sequence = PDBparser.get_sequence()
        pdbresID = {chain: list(PDBparser.get_pdbresID2absID(chain).keys())}
    except Exception as e:
        import traceback;traceback.print_exception(e)
        import pdb; pdb.set_trace()
    data_dict = {
        'atom_bb': atom_bb,
        'atom_37': atom_37,
        'sequence': sequence,
        'pdbres_idx': pdbresID
    }

    return data_dict
    # except:
    #     return None


def cand_lig_select(coords37, ligcoords, min_dist=5.0):
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
    parser.add_argument('--out_dir', type=str)
    parser.add_argument("--protein_chain", type=str, default='A')
    parser.add_argument("--atomized_chain", type=str, default='')
    parser.add_argument("--merge_atomized_chain", action='store_true', default=False)

    args = parser.parse_args()
    return args



def make_prot_lig_complex_from_pdb(args):
    pdb_dir = args.pdb_dir
    outdir = args.out_dir
    protein_chain = args.protein_chain
    atomized_chain = args.atomized_chain
    if (len(atomized_chain) == 0):
        atomized_chain = None

    for pdb_f in os.listdir(pdb_dir):
        prot_lig_pdb_f = f'{pdb_dir}/{pdb_f}'
        if prot_lig_pdb_f.endswith('.pdb'):
            prot_lig_pdbname = os.path.basename(prot_lig_pdb_f).split('.pdb')[0]
        elif prot_lig_pdb_f.endswith('.cif'):
            prot_lig_pdbname = os.path.basename(prot_lig_pdb_f).split('.cif')[0]
        else:
            prot_lig_pdbname = os.path.basename(prot_lig_pdb_f)
        npy_f = f'{outdir}/all_npy/{prot_lig_pdbname}.npy'
        os.makedirs(f'{outdir}/all_npy', exist_ok=True)
        lig_dir = f'{outdir}/all_lig/{prot_lig_pdbname}'
        os.makedirs(lig_dir, exist_ok=True)
        process(prot_lig_pdb_f, lig_dir, atomized_chain, args.merge_atomized_chain)
        lig_name_list = list(set([lig_name_f.split('.')[0] for lig_name_f in os.listdir(lig_dir)]))

        prot_data = get_npy_from_cif(prot_lig_pdb_f, protein_chain)
        if (prot_data is None):
            print(f'{prot_lig_pdb_f} {protein_chain} load failed')
            continue

        prot_lig_complex_dict = {}

        prot_lig_complex_dict['prot'] = prot_data
        prot_lig_complex_dict['lig'] = {}

        for lig_name in lig_name_list:
            lig_pdbcode = lig_name.split('_')[2]
            if lig_pdbcode in ligMPNN_ignore_mol_type:
                continue

            mol = None
            for lig_data_mode in cand_lig_f_type:
                absl_lig_f = f'{lig_dir}/{lig_name}.{lig_data_mode}'

                mol = read_molecule(absl_lig_f, remove_hs=True, sanitize=True)
                if (mol is not None):
                    break

            if (mol is None):
                print(lig_name, 'unloaded')
                continue

            mol = read_molecule(absl_lig_f,remove_hs=True, sanitize=True)
            lig_coords = mol.GetConformer().GetPositions()
            prot_coords37 = prot_data['atom_37']
            lig_flag = cand_lig_select(prot_coords37, lig_coords, min_dist=5.0)

            if lig_flag:
                cur_lig_graph = get_lig_graph(mol)
                unimol_reprs_dict = get_unimol_repr_from_pos(cur_lig_graph['ligand'], mol)

                if (cur_lig_graph['ligand'].pos.shape[0] == unimol_reprs_dict['atomic_reprs'][0].shape[0]):
                    prot_lig_complex_dict['lig'][f'{lig_name}.{lig_data_mode}'] = {'data': cur_lig_graph, 'rdkit_ligand': mol, 'unimol_reprs': unimol_reprs_dict}
                print(f'{lig_name}.{lig_data_mode} loaded')
            else:
                print(f'{lig_name}.{lig_data_mode} ignored')

        np.save(npy_f, prot_lig_complex_dict)
        print(f'{prot_lig_pdb_f} {protein_chain} preprocessed')


if __name__ == '__main__':
    args = get_args()
    make_prot_lig_complex_from_pdb(args)

    # prot_lig_pdb_f = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/protenc_ligenc_enc_dec_pifold_all_lig_preAAenc/test_data/merged_pdb_dir/ifp_from_hxh.cif'
    # chain = 'A'
    # outdir ='/raw22/superbrain/permanent/yfliu25/ligand_protdesign/protenc_ligenc_enc_dec_pifold_all_lig_preAAenc/test_data/npy_dir'
    # make_prot_lig_complex_from_pdb(prot_lig_pdb_f, chain, outdir)
