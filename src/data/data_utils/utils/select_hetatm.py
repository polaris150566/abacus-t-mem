from Bio import __version__
#print('BIOPYTHON Version : ' , __version__)
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBList, PDBParser
from Bio.PDB.mmcifio import MMCIFIO

from rdkit.Chem import AllChem
import pdb

import subprocess
import numpy as np
import os

from converter import convert_file

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

class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.id == self.chain.id:
            return True #1
        else:
            return False #0

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return True #1


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
def process(filename, outdir, atomized_chain=None, merge_atomized_chain=False):
    """
    对于输入的filename文件(PDB/CIF),将其解析之后，提取第一个模型(PDB通常也只有一个模型),
    之后将其配体转换为（如 PDB、Mol2、SDF、PDBQT)类型文件并存贮

    如果 merge_atomized_chain为true则将整个肽链作为一个分子对象统一存储


    Args:
        filename (_type_):输入文件夹
        outdir (_type_): 输出文件夹
        atomized_chain (_type_, optional): 如果为true则可能不进行任何输出_. Defaults to None.
        merge_atomized_chain (bool, optional): 如果为是，则将所有的链作为一个整体进行输出
    """
    name = '.'.join(os.path.basename(filename).split('.')[:-1])
    try:
        pars = MMCIFParser(QUIET = True)
        #print(name, filename, flush=True)
        struct = pars.get_structure(name,  filename)

        # pdb.set_trace()
    except:
        pars = PDBParser(QUIET = True)
        struct = pars.get_structure(name,  filename)

    model = list(struct.child_dict.values())[0]

    if ((atomized_chain is not None) and merge_atomized_chain):
        io = PDBIO()
        io.set_structure(struct)

    i = 1
    for chain in model:
        if atomized_chain is not None:
            if str(chain).split('=')[-1][:-1] == atomized_chain:
                atomizing = True
            else:
                atomizing = False
            atomize_aa = True
        else:
            atomizing = True
            atomize_aa = False

        if atomizing:
            if merge_atomized_chain:
                io = PDBIO()
                io.set_structure(struct)
                pdb_path = f'{outdir}/lig_{chain.id}_PEP_{i}_{i}.pdb'
                mol_path = f'{outdir}/lig_{chain.id}_PEP_{i}_{i}.mol2'
                sdf_path = f'{outdir}/lig_{chain.id}_PEP_{i}_{i}.sdf'
                pdbqt_path = f'{outdir}/lig_{chain.id}_PEP_{i}_{i}.pdbqt'
                io.save(pdb_path, ChainSelect(chain))

                convert_file(pdb_path, sdf_path)
                convert_file(pdb_path, mol_path)
                convert_file(pdb_path, pdbqt_path)

            else:
                for res in chain:
                    # import pdb;pdb.set_trace()
                    if not atomize_aa:
                        if (res.resname in restype3_list) and res.get_full_id()[3][0] == ' ':#筛掉所有正常的氨基酸
                            # print("natural !")
                            # import pdb;pdb.set_trace()
                            continue

                    if res.resname=='HOH':
                        continue

                    io = PDBIO()
                    io.set_structure(struct)
                    os.makedirs(outdir,exist_ok=True)
                    raw_resname, pdbres_i, seg_i = res.id
                    pdb_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.pdb'
                    mol_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.mol2'
                    sdf_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.sdf'
                    pdbqt_path = f'{outdir}/lig_{chain.id}_{res.resname}_{pdbres_i}_{i}.pdbqt'

                    try:
                        io.save(pdb_path, ResidueSelect(chain, res))
                    except Exception as excp:
                        continue

                    convert_file(pdb_path, sdf_path)
                    convert_file(pdb_path, mol_path)
                    convert_file(pdb_path, pdbqt_path)
                    i += 1
def multi(path):
    try:
        process(path)
    except:
        pass