from Bio import __version__
print('BIOPYTHON Version : ' , __version__)
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBList
import subprocess
import numpy as np
import os

lig_pdb_dir = '/database/lyf_database/all_pdb20230101_20240112_lig'

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
non_standardAA = {"ASX": "D", "XAA": "G", "GLX": "E", "XLE": "L", "SEC": "C", "PYL": "K", "UNK": "G", "PTR": "Y", "MSE": "M"}

restype3_list = list(restype_1to3.values()) + ['UNK'] + list(non_standardAA.keys())


def is_het(residue):
    res = residue.resname 

    if len(res) ==1 and res in  ["A","C",'G','T','U']:
        return True #1
    else:
        return False #0

class ResidueSelect(Select):
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        if chain.id == self.chain.id:
            return True #1
        else:
            return False #0
        
    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        if residue == self.residue:
            return True #1
        else:
            return False #0

def delete_empty(data_f):
    with open(data_f, 'r') as reader:
        all_lines = reader.readlines()

    if data_f.endswith('.pdb'):
        all_line = ''.join(all_lines)
        if not ( ('ATOM' in all_line) or ('HETATM' in all_line) ):
            os.system(f'/bin/rm -rf {data_f}')
    else:
        if (len(all_lines) == 0):
            os.system(f'/bin/rm -rf {data_f}')


# for name in names:
def process(name):
    outdir = f'{lig_pdb_dir}/{name[1:3]}/{name}'
    filename = f'/database/lyf_database/pdb_update/pdb_20230101_20240112/{name[1:3]}/{name}.cif'
    pars = MMCIFParser(QUIET = True)
    try:
        struct = pars.get_structure(name,  filename)
    except:
        return

    model = list(struct.child_dict.values())[0]
    i = 1
    for chain in model:
        for res in chain:
            if (res.resname not in restype3_list) :
                if res.resname=='HOH':
                    continue
                # print('\n'+'res : ', res, res.id, res.resname)
                io = PDBIO()
                io.set_structure(struct)
                os.makedirs(outdir,exist_ok=True)
                raw_resname, pdbres_i, seg_i = res.id
                pdb_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.pdb'
                mol_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.mol2'
                sdf_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.sdf'

                try:
                    io.save(pdb_path, ResidueSelect(chain, res))
                except:
                    continue
                
                with open(pdb_path, 'r') as reader:
                    all_lines = reader.readlines()
                    
                all_line = ''.join(all_lines)
                if not ( ('ATOM' in all_line) or ('HETATM' in all_line) ):
                    os.system(f'/bin/rm -rf {pdb_path}')
                    continue
                
                try:
                    subprocess.run(["/home/liuyf/cpp_bin/ob/bin/babel", pdb_path, sdf_path])
                except Exception as excp:
                    pass
                try:
                    subprocess.run(["/home/liuyf/cpp_bin/ob/bin/babel", pdb_path, mol_path])
                except Exception as excp:
                    pass
                i += 1
                

from joblib import Parallel,delayed
from tqdm import tqdm

import glob
names = glob.glob('/database/lyf_database/pdb_update/pdb_20230101_20240112/*/*')
names = [os.path.basename(name).split('.cif')[0] for name in names]

for path in tqdm(names,total=len(names)):
    process(path, )
    import pdb; pdb.set_trace()
    
# names = [os.path.basename(name).split('.cif')[0] for name in names]
# Parallel(32)(delayed(process)(path) for path in tqdm(names,total=len(names)))

# ensemble_cluster_f = '/raw22/superbrain/permanent/yfliu25/dataset/vq_monomer_data/utils/avail_allosteric_fa_dict.npy'
# all_lig_dir = '/train14/superbrain/lili27/protein/full_atom_protein_with_molecular_work/data_processing/processing/ligs'


# ensemble_cluster_dict = np.load(ensemble_cluster_f, allow_pickle=True).item()
# all_allo_cluster_prot_chain_list = []
# all_allo_pdbcode_list = []

# for rep_prot_chain, mem_prot_chain_ctx in ensemble_cluster_dict.items():
#     mem_prot_chain_list = mem_prot_chain_ctx['rep']
#     all_allo_cluster_prot_chain_list.extend(mem_prot_chain_list)
    
# for allo_prot_chain in all_allo_cluster_prot_chain_list:
#     pdbcode, prot_chain, model_id, exp_type = allo_prot_chain.split('_')
#     if (exp_type == 'cry'):
#         all_allo_pdbcode_list.append(pdbcode)

# names = sorted(list(set(all_allo_pdbcode_list)))

# import glob
# names = glob.glob('/database/lyf_database/pdb_update/pdb_20230101_20240112/*/*')
# # root = '/raw22/superbrain/permanent/yfliu25/dataset/vq_monomer_data/all_pdb_lig'


# # root = glob.glob('/raw22/superbrain/permanent/yfliu25/dataset/vq_monomer_data/all_pdb_lig/*/*/*')
# # np.save('tmp_all_lig_absl_path.npy', root)
# root = np.load('tmp_all_lig_absl_path.npy', allow_pickle=True).tolist()
# Parallel(32)(delayed(delete_empty)(path) for path in tqdm(root,total=len(root)))

