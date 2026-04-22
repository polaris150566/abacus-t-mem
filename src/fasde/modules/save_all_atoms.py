import numpy as np
# import sys
# sys.path.append('../alphafold2')
# from fasde.utils.relax.assess_violation import get_violation_metrics
from fasde.modules.alphafold.common import protein
# from alphafold.common import protein

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
    if (file_path is not None):
        with open(file_path, 'w') as wf:
            wf.write(pdb_str)
    return pdb_str
   

def write_coords(atom37, aatype, atom_mask, out_pdb_f=None):
    len_ = ((atom_mask).sum(-1)!=0).sum().item()
    atom_coords = atom37[:len_].cpu().data.numpy()
    mask = atom_mask[:len_].float().cpu().data.numpy()
    aatype = aatype[:len_].cpu().data.numpy()
    res_index = np.arange(len_)+1
    chain_index = np.zeros(len_, dtype=int)
    rec_str = save_to_pdb(out_pdb_f,atom_coords,mask,aatype,res_index,chain_index)

    return rec_str