"""
Pre-compute per-residue relative SASA for all train+valid samples.
Uses multiprocessing; each worker parses one .ent file and returns
  { mf_basename: np.array([relative_sasa per residue]) }
Residues are matched by (chain_id, auth_seq_num) against merged_npy's
pdb_chain_mask and pdbres_idx, avoiding positional slice errors.

Output: /home/chenty/abacust_mem/auxiliary_test/interface_probe/sasa_cache.npy
"""
import os, sys, time, multiprocessing as mp
import numpy as np
from collections import defaultdict

MERGED_DIR = '/home/chenty/abacust_mem/src/data/data_storage/merged_npys/all_npy/'
PDB_DIR    = '/home/chenty/abacust_mem/src/data/data_storage/assembled_pdbs/'
CLUSTER_NPY= '/home/chenty/abacust_mem/src/data/data_storage/merged_cluster_dict.npy'
OUT_PATH   = '/home/chenty/abacust_mem/auxiliary_test/interface_probe/sasa_cache.npy'
N_WORKERS  = 16

MAX_SASA = {
    'ALA':121.,'CYS':148.,'ASP':187.,'GLU':214.,'PHE':228.,'GLY':97.,
    'HIS':216.,'ILE':195.,'LYS':230.,'LEU':191.,'MET':203.,'ASN':187.,
    'PRO':154.,'GLN':214.,'ARG':265.,'SER':143.,'THR':163.,'VAL':165.,
    'TRP':264.,'TYR':255.,
}
DEFAULT_MAX_SASA = 200.0


def process_one(args):
    """Worker: parse one .ent, compute SASA, match residues by auth_seq_num."""
    stem, mf_list = args
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    ent_path = os.path.join(PDB_DIR, stem + '.ent')
    if not os.path.exists(ent_path):
        return {}

    try:
        parser = PDBParser(QUIET=True)
        sr = ShrakeRupley()
        struct = parser.get_structure('p', ent_path)
        sr.compute(struct[0], level='R')

        # chain_sasa: {chain_id: {auth_seq_num: relative_sasa}}
        chain_sasa = {}
        for chain in struct[0]:
            d = {}
            for res in chain:
                if res.id[0] != ' ':   # skip HETATM
                    continue
                auth_seq = res.id[1]   # PDB author sequence number
                sasa = getattr(res, 'sasa', 0.0)
                max_s = MAX_SASA.get(res.resname, DEFAULT_MAX_SASA)
                d[auth_seq] = float(sasa) / max_s if max_s > 0 else 0.0
            chain_sasa[chain.id] = d
    except Exception:
        return {}

    result = {}
    for mf in mf_list:
        try:
            merged = np.load(os.path.join(MERGED_DIR, mf), allow_pickle=True).item()
            seq           = merged['prot']['sequence']
            chain_mask    = merged['prot']['pdb_chain_mask']   # [L] array of chain ids
            pdbres_idx = merged['prot']['pdbres_idx']    # [L] array of auth seq nums

            rel_sasa = np.zeros(len(seq), dtype=np.float32)
            for i, (ch, auth_seq) in enumerate(zip(chain_mask, pdbres_idx)):
                rel_sasa[i] = chain_sasa.get(ch, {}).get(int(auth_seq), 0.0)

            result[mf] = rel_sasa
        except Exception:
            pass

    return result


def main():
    cluster_dict = np.load(CLUSTER_NPY, allow_pickle=True).item()
    all_members = set()
    for split in ('train', 'valid'):
        for ms in cluster_dict[split].values():
            all_members.update(ms)

    merged_files = [f for f in os.listdir(MERGED_DIR) if f.endswith('.npy')]
    code_to_mfiles = defaultdict(list)
    for f in merged_files:
        code_to_mfiles[f[:4].lower()].append(f)

    target_files = set()
    for name in all_members:
        for mf in code_to_mfiles.get(name[:4].lower(), []):
            target_files.add(mf)

    # group by ent stem
    ent_to_samples = defaultdict(list)
    for mf in target_files:
        stem = mf.rsplit('_', 1)[0] + '_'
        ent_to_samples[stem].append(mf)

    jobs = list(ent_to_samples.items())
    print(f"Target files: {len(target_files)}, unique .ent: {len(jobs)}, workers: {N_WORKERS}")

    cache = {}
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for i, partial in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4)):
            cache.update(partial)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(jobs) - i - 1)
                print(f"  [{i+1}/{len(jobs)}] cached={len(cache)} ETA={eta/60:.1f}min")

    print(f"Done. Cached {len(cache)} / {len(target_files)} samples. Saving...")
    np.save(OUT_PATH, cache)
    print(f"Saved to {OUT_PATH}")


if __name__ == '__main__':
    main()
