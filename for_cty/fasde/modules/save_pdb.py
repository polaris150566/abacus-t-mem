import numpy as np
import sys
sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
from alphafold.common import protein

def save_to_pdb(file_path,all_atoms,all_mask,aatype,res_index,chain_index):
    bfactor = np.zeros_like(all_mask)
    prot = protein.Protein(
        atom_positions=all_atoms,
        atom_mask=all_mask,
        aatype=aatype,
        residue_index=res_index,
        chain_index=chain_index,
        b_factors = bfactor
        )
    pdb_str = protein.to_pdb(prot)

    with open(file_path, 'w') as wf:
        wf.write(pdb_str)