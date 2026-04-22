import os, sys
import numpy as np
np.set_printoptions(linewidth=1000)

def run(fa_f):
    with open(fa_f, 'r') as reader:
        all_lines = reader.readlines()
    
    seq_dict = {}
    for l_idx, line in enumerate(all_lines):
        if line.startswith('>'):
            query = line[1:].strip()
            seq = all_lines[l_idx+1].strip()
            seq_dict[query] = seq

    identity_mat = np.zeros((len(seq_dict), len(seq_dict)))
    for q_idx, (query, q_seq) in enumerate(seq_dict.items()):
        for j_idx, (query, j_seq) in enumerate(seq_dict.items()):
            ident = (np.array(list(q_seq)) == np.array(list(j_seq))).sum()/len(j_seq)
            identity_mat[q_idx, j_idx] = round(ident, 3)
            # identity_mat[q_idx, j_idx] = (np.array(list(q_seq)) == np.array(list(j_seq))).sum()

        
    import pdb; pdb.set_trace()
    # (1 - identity_mat).sum(-1)
    (1 - identity_mat[1:, 1:]).sum(-1)


if __name__ == '__main__':
    # fa_f = 'esm_3B_4pn2.fasta'
    # fa_f = 'msa_4pn2.fasta'
    # fa_f = '4pn2_exp.fasta'
    # fa_f = 'allose_ligand_exp.fasta'
    # fa_f = '1jvj_3B_MSA_candidate.fa'
    # fa_f = '1lvm_3B_msa_candidate.fa'
    fa_f = 'allose_3B_exp.fasta'
    run(fa_f)