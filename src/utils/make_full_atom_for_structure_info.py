
import logging
import os
import sys
from multiprocessing import Pool
import time
from typing import List, Tuple, Sequence
from copy import deepcopy

import numpy as np

from protein_map_gen import FastPoteinParser

WORKERS = 30


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)



restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.


def get_protein_list_from_cath(file: str):
    protein_list = []
    with open(file, "r") as reader:
        for line in reader.readlines():
            protein_list.append(line.strip().split()[0])

    return protein_list


def get_protein_list(file: str):
    protein_list = []
    with open(file, "r") as reader:
        for line in reader.readlines():
            if len(line.strip()) > 0:
                protein_list.append(line.strip())
            else:
                continue

    return protein_list


def nodewraprun(mmCIF_dir, protein_chain, out_dir, protein, logfile):
    try:
        print(f"protein: {protein} is processing ...")
        # /ps2/hyperbrain/yfliu25/cath_dataset/split_data/r2/1r26A00A/1r26A00.pdb
        # import pdb; pdb.set_trace()
        tmp_parroot = os.path.join(out_dir, protein)
        
        if not os.path.isdir(tmp_parroot):
            os.mkdir(tmp_parroot)

        data_file = os.path.join(tmp_parroot, "PDBstructure_data.npz")
        # import pdb; pdb.set_trace()
        add_all_atoms(
            pdbfile=mmCIF_dir, filetype="cif",
            chain=protein_chain, data_file=data_file)
        print(data_file, 'add sidechain info')

    except FileNotFoundError:
        with open(logfile, "a") as writer:
            writer.write(f'fail preprocess protein: {protein}')
            writer.write("\n")


def add_all_atoms(pdbfile, filetype, chain, data_file):

    PDBparser = FastPoteinParser(poteinfile=pdbfile, chain=chain, 
                                datatype=filetype, mask_dist=False )
    all_atom_position_dict = PDBparser.chain_crd_dicts
    # resnum = len(all_atom_position_dict.keys())
    all_atom_positions = np.asarray([list(all_atom_position_dict[k].values())[:-1] for k in all_atom_position_dict.keys()])
    all_atom_mask = np.all(all_atom_positions != 0, -1).astype(np.int32)
    sequence = PDBparser.sequence
    encoded_sequence = np.asarray([restype_order[aa] for aa in sequence])

    data_dict = {}
    data_dict['full_atoms'] = {
        'aatype': encoded_sequence,
        'all_atom_positions': all_atom_positions,
        'all_atom_mask': all_atom_mask
    }
    # import pdb; pdb.set_trace()
    np.savez(data_file, data_dict)


def multiprocess_run(proteinfile: str, out_root: str, mmCIF_root: str,
                     merge_dssp_=False, dssp_root=None, logfile=None,
                     k=None, overrun=True, overwrite=True):

    if merge_dssp_: assert dssp_root is not None
    protein_list = get_protein_list_from_cath(proteinfile)

    if os.path.isdir(out_root): pass
    else: os.mkdir(out_root)

    pool = Pool(WORKERS)

    for protein in protein_list:
        protein_name = protein
        protein_chain = protein[4]

        mmCIF_dir = os.path.join(mmCIF_root, protein_name[1:3], protein_name[:4]+".cif")

        if merge_dssp_: dssp_dir = os.path.join(dssp_root, protein_name[1:3], protein_name[:4]+".dssp")
        else: dssp_dir = None

        out_dir = os.path.join(out_root, protein_name[1:3])
        if os.path.isdir(out_dir): pass
        else: os.mkdir(out_dir)

        # if not overrun:
        #     assert (overwrite==True)
        #     if not os.path.isfile(os.path.join(out_dir, protein, "PDBstructure_data.npz")):
        #         import pdb; pdb.set_trace()

        # nodewraprun(mmCIF_dir, protein_chain, out_dir, protein, logfile)

        pool.apply_async(func=nodewraprun,
                         args=(mmCIF_dir, protein_chain, out_dir, protein, logfile, ))

    pool.close()
    pool.join()


if __name__ == "__main__":
    OUT_ROOT = "/train14/superbrain/lhchen/for_yfliu25/all_atom_local_sd"
    # OUT_ROOT = "/ps2/hyperbrain/yfliu25/local_sd"
    # mmCIF_ROOT = "/ps2/hyperbrain/yfliu25/cath_dataset/split_data"
    mmCIF_ROOT = "/train14/superbrain/lhchen/data/PDB/20220102/mmcif"
    DSSP_ROOT = "/ps2/hyperbrain/yfliu25/cath_dataset/split_dssp"
    logfile = "/train14/superbrain/lhchen/for_yfliu25/all_atoms.log"
    # protein_file = "/yrfs1/hyperbrain/yfliu25/dataset/testcath"
    protein_file = "/train14/superbrain/yfliu25/dataset/cath-b-newest-all"
    # protein_file = "/yrfs1/hyperbrain/yfliu25/confGF/ConfGF_CATH_Rigid_SCUBASD/data/list_with_len/test_10-cath-b-all"
    # protein_file = "/home/liuyf/proteins/DDPM_loop/no_restriction_loop/utils/test_1r26.txt"


    time0 = time.time()
    multiprocess_run(proteinfile=protein_file, out_root=OUT_ROOT, mmCIF_root=mmCIF_ROOT,
                     dssp_root=DSSP_ROOT, merge_dssp_=False, logfile=logfile, k=1000, overrun=False, overwrite=True)
    time1 = time.time()
    print(time1-time0)

    # for _ in range(1000):
    #     time0 = time.time()
    #     multiprocess_run(proteinfile=protein_file, out_root=OUT_ROOT, mmCIF_root=mmCIF_ROOT,
    #                     dssp_root=DSSP_ROOT, merge_dssp_=False, logfile=logfile, random_temp=True, 
    #                     highest_temp=5, lowest_temp=0.1, trajstep=100, timestep=0.002, totalstep=2000,
    #                     temp_step_num=15, k=1000, gamma=7.0, overrun=True)
    #     time1 = time.time()
    #     print(time1-time0)