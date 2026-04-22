import os, sys
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from .unimol_tools import UniMolRepr

from torch.utils.data import DataLoader, Dataset


# from func_timeout import func_set_timeout
# import func_timeout

max_parse_time = 40
# single smiles unimol representation
clf = UniMolRepr(data_type='molecule', remove_hs=True, use_gpu=False)

# @func_set_timeout(max_parse_time)
def get_unimol_repr_from_pos(lig_ctx, mol_from_mol):
    mol_from_mol = Chem.RemoveHs(mol_from_mol)
    atoms_list = [atom.GetSymbol() for atom in mol_from_mol.GetAtoms()]
    coords_list = (lig_ctx.pos).numpy()
    input_data = {'atoms': atoms_list, 'coordinates': coords_list}
    mol_from_smiI_rep = clf.get_repr(input_data, return_atomic_reprs=True)

    return mol_from_smiI_rep


# @func_set_timeout(max_parse_time)
def get_unimol_repr(mol_from_mol):
    mol_from_mol = Chem.RemoveHs(mol_from_mol)
    smiles = Chem.MolToSmiles( mol_from_mol, isomericSmiles=True )
    mol_from_smiI = Chem.MolFromSmiles(smiles)    # create the same molecule from the provided Inchified SMILES
    print('mol preprocess')
    assert(mol_from_smiI.GetNumAtoms() == mol_from_mol.GetNumAtoms())
    match = mol_from_smiI.GetSubstructMatch(mol_from_mol)    
    assert(match)
    assert(len(match) == mol_from_smiI.GetNumAtoms())
    # renumbered_mol_from_smiI = Chem.RenumberAtoms(mol_from_smiI, match)
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=True)
    smiles_list = [smiles]
    mol_from_smiI_rep = clf.get_repr(smiles_list, return_atomic_reprs=True)
    renumbered_atomic_reprs = np.array(mol_from_smiI_rep['atomic_reprs'])[0][np.array(match)]
    renumbered_atomic_symbol = np.array(mol_from_smiI_rep['atomic_symbol'])[0][np.array(match)].tolist()
    mol_from_smiI_rep['renumbered_atomic_reprs'] = renumbered_atomic_reprs
    mol_from_smiI_rep['renumbered_atomic_symbol'] = renumbered_atomic_symbol

    return mol_from_smiI_rep


def run(npy_dir, new_dir):
    
    def collated_fn(batch):
        b_process = []
        for b in batch:
            if b:
                cur_process_f = b
                b_process.append(cur_process_f)
                
        return b_process
    
    class PredictionDataset(Dataset):
        def __init__(self, npy_dir, new_dir) -> None:
            super().__init__()
            self.pdbname_list = []
            for subdir in os.listdir(npy_dir):
                self.pdbname_list.extend([
                    (f'{npy_dir}/{subdir}/{f_}', f'{new_dir}/{subdir}/{f_}') \
                        for f_ in os.listdir(f'{npy_dir}/{subdir}')])
     
        def __getitem__(self, index):
            absl_old_npy_f, absl_new_npy_f = self.pdbname_list[index]
            try:
                data = np.load(absl_old_npy_f, allow_pickle=True).item()
                mol = data['rdkit_ligand']
                # unimol_reprs_dict = get_unimol_repr(mol)
                unimol_reprs_dict = get_unimol_repr_from_pos(data['complex']['ligand'], mol)
                data['unimol_reprs'] = unimol_reprs_dict
                os.makedirs(os.path.dirname(absl_new_npy_f), exist_ok=True)
                np.save(absl_new_npy_f, data)
                print(absl_new_npy_f)
                return True
            except IndexError:
                return False
            except AttributeError:
                return False
            except AssertionError:
                return False
            except func_timeout.exceptions.FunctionTimedOut:
                print(f'parsing {absl_new_npy_f} timeout')
                return False
                
        def __len__(self,):
            return len(self.pdbname_list)

    pdb_align_dataset = PredictionDataset(npy_dir, new_dir)
    pdb_align_dataloader = DataLoader(pdb_align_dataset, batch_size=20, num_workers=40, collate_fn=collated_fn)
    
    for batch_fa_dict in tqdm(pdb_align_dataset):
        pass

if __name__ == '__main__':
    # npy_dir = '/raw22/superbrain/permanent/yfliu25/dataset/prot_with_ligands_v2'
    npy_dir = '/raw22/superbrain/permanent/yfliu25/dataset/debug'
    # new_dir = '/raw22/superbrain/permanent/yfliu25/dataset/prot_with_ligands_unimol'
    new_dir = '/raw22/superbrain/permanent/yfliu25/dataset/debug_raw_out'
    run(npy_dir, new_dir)