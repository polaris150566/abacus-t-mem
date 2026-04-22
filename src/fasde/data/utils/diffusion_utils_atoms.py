import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import random
import logging
import copy

from torch_geometric.utils import to_dense_adj
from fairseq.data import data_utils
#from .bonds import make_angle_torsion_index
from bonds import make_angle_torsion_index

# import sys
# sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
# from alphafold import all_atom
# from fasde.modules.alphafold  import all_atom
# sys.path.append('/raw22/superbrain/permanent/yfliu25/unimol/Uni-Mol-main/unimol_tools')
# from unimol_tools import UniMolRepr
# clf = UniMolRepr(data_type='molecule', remove_hs=True, use_gpu=False)

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_order_to_name = {i: name for name, i in restype_order.items()}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

fake_lig_restypes = [
    'R', 'N', 'D', 'Q', 'E', 'H', 'L', 'K', 'M', 'F', 'W', 'Y'
]
fake_lig_restypes_id = [restype_order[restype] for restype in fake_lig_restypes]

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


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}


logger = logging.getLogger(__file__)


# single_res_lig_graph_dir = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/aatype_mol_feature/fake_model_feature'
# res_lig_graph_dict = {}
# for lig_graph_f in os.listdir(single_res_lig_graph_dir):
#     aatype_name = lig_graph_f.split('_')[0]
#     absl_lig_graph_f = f'{single_res_lig_graph_dir}/{lig_graph_f}'
#     try:
#         res_lig_graph_ctx = np.load(absl_lig_graph_f, allow_pickle=True).item()
#     except:
#         import pdb; pdb.set_trace()
#     res_lig_graph_dict[aatype_name] = res_lig_graph_ctx


# single_res_lig_graph_unimol_dir = '/raw22/superbrain/permanent/yfliu25/ligand_protdesign/aatype_mol_feature/fake_model_feature_unimol'
# res_lig_graph_unimol_dict = {}
# for lig_graph_f in os.listdir(single_res_lig_graph_unimol_dir):
#     aatype_name = lig_graph_f.split('_')[0]
#     absl_lig_graph_unimol_f = f'{single_res_lig_graph_unimol_dir}/{lig_graph_f}'
#     try:
#         res_lig_graph_ctx = np.load(absl_lig_graph_unimol_f, allow_pickle=True).item()
#     except:
#         import pdb; pdb.set_trace()
#     res_lig_graph_unimol_dict[aatype_name] = res_lig_graph_ctx
UNIMOL_REPRS_DIM=512


def seq_given_mask(ca_coord, lig_pos, ca_radius=9):
    ca_coord = torch.FloatTensor(ca_coord)

    dist = torch.sqrt(torch.sum((ca_coord[:, None] - lig_pos[None])**2, -1)).min(-1)[0] # P, L, 3
    masked_seq_res = dist < ca_radius
    return masked_seq_res


def make_ligand_node_feature(node_attr):
    #from ...modules.mols.process_mols import allowable_features
    sys.path.append('/home/chenty/abacust_mem/src/fasde/modules/mols')
    from process_mols import allowable_features
    feature_types = [
        'possible_atomic_num_list',
        'possible_chirality_list',
        'possible_degree_list',
        'possible_formal_charge_list',
        'possible_implicit_valence_list',
        'possible_numH_list',
        'possible_number_radical_e_list',
        'possible_hybridization_list',
        'possible_is_aromatic_list',
        'possible_numring_list',
        'possible_is_in_ring3_list',
        'possible_is_in_ring4_list',
        'possible_is_in_ring5_list',
        'possible_is_in_ring6_list',
        'possible_is_in_ring7_list',
        'possible_is_in_ring8_list',
    ]

    feature_sizes = [len(allowable_features[ftype]) for ftype in feature_types]

    num_features = len(feature_types)
    assert(node_attr.shape[-1] == num_features)

    onehot_feature = []
    for i in range(num_features):
        featidx = node_attr[..., i]
        feat_ = F.one_hot(featidx, num_classes=feature_sizes[i])

        onehot_feature.append(feat_)

    onehot_feature = torch.cat(onehot_feature, dim=-1)
    return onehot_feature


def make_fake_lig_pos_from_atom37(start_res_idx_list, all_atom_positions, all_atom_mask, aatype, make_unimol_feature=False):
    start_fake_lig_idx = 5
    node_sum = 0
    node_attr = []
    edge_index = []
    edge_attr = []
    lig_pos = []
    if make_unimol_feature:
        unimol_reprs = []
    for start_res_idx in start_res_idx_list:
        res_aatype_3 = restype_1to3[restype_order_to_name[aatype[start_res_idx].item()]]
        if make_unimol_feature:
            res_graph = res_lig_graph_unimol_dict[res_aatype_3]
        else:
            res_graph = res_lig_graph_dict[res_aatype_3]
        cur_fake_res_node_attr = res_graph['ligand']['ligand'].x
        cur_fake_res_num = cur_fake_res_node_attr.shape[0]
        cur_fake_res_edge_attr = res_graph['ligand']['ligand', 'lig_bond', 'ligand'].edge_attr
        cur_fake_res_edge_index = res_graph['ligand']['ligand', 'lig_bond', 'ligand'].edge_index

        edge_index.append(cur_fake_res_edge_index + node_sum)
        edge_attr.append(cur_fake_res_edge_attr)
        node_attr.append(cur_fake_res_node_attr)

        res_atom37_coords = all_atom_positions[start_res_idx]
        res_atom37_mask = all_atom_mask[start_res_idx]
        fake_lig_coords = res_atom37_coords[res_atom37_mask.bool()][start_fake_lig_idx:]
        lig_pos.append(fake_lig_coords)

        if make_unimol_feature:
            unimol_reprs.append(torch.from_numpy(res_graph['unimol_reprs']))

        node_sum += cur_fake_res_num

    node_attr = torch.cat(node_attr)
    edge_index = torch.cat(edge_index, -1)
    edge_attr = torch.cat(edge_attr)
    lig_pos = torch.cat(lig_pos)
    assert lig_pos.shape[0] == node_attr.shape[0]
    lig_single = make_ligand_node_feature(node_attr)
    if make_unimol_feature:
        unimol_reprs = torch.cat(unimol_reprs)
    #     atoms_list = [atom.GetSymbol() for atom in mol_from_mol.GetAtoms()]
    #     coords_list = (lig_ctx.pos).numpy()
    #     input_data = {'atoms': atoms_list, 'coordinates': coords_list}
    #     mol_from_smiI_rep = clf.get_repr(input_data, return_atomic_reprs=True)
    if make_unimol_feature:
        return lig_single, edge_index, edge_attr, lig_pos, unimol_reprs
    else:
        return lig_single, edge_index, edge_attr, lig_pos



def make_triple_edges(edge_index, edge_mask):
    edge_index = edge_index.T

    neighbours = {}
    for st, ed in edge_index.numpy():
        if st not in neighbours:
            neighbours[st] = []
        if ed not in neighbours:
            neighbours[ed] = []
        if ed not in neighbours[st]:
            neighbours[st].append(ed)
        if st not in neighbours[ed]:
            neighbours[ed].append(st)

    tor_edge_index = edge_index[edge_mask]
    tor_edge_mask = torch.ones((tor_edge_index.shape[0])).float()

    prev_st, next_ed = [], []
    for st, ed in tor_edge_index.numpy():
        idx = np.random.permutation(len(neighbours[st]))
        prev_ = neighbours[st][idx[0]] if neighbours[st][idx[0]] != ed else neighbours[st][idx[1]]
        idx = np.random.permutation(len(neighbours[ed]))
        next_ = neighbours[ed][idx[0]] if neighbours[ed][idx[0]] != st else neighbours[ed][idx[1]]

        prev_st.append(prev_)
        next_ed.append(next_)

    prev_edge = torch.LongTensor(prev_st)
    next_edge = torch.LongTensor(next_ed)
    tor_edge_index = torch.cat([prev_edge[..., None], tor_edge_index, next_edge[..., None]], dim=-1)
    return tor_edge_index, tor_edge_mask


def make_lig_features(data):
    pos = data['ligand']['pos']
    node_feature = data['ligand']['x']

    edge = data['ligand', 'lig_bond', 'ligand']
    edge_index = edge.edge_index
    edge_attr = edge.edge_attr

    single_feature = make_ligand_node_feature(node_feature)
    edge_attr = data_utils.pad_to_length(edge_attr + 1, 5, 1, 1)
    pair_feature = to_dense_adj(edge_index, edge_attr = edge_attr)[0]

    return single_feature, pair_feature


def make_complex_feature(
    rec_feature, raw_data,
    lig_single_dim, lig_pair_dim,
    sde,
    t=None,
    num_aatypes=22,
    fix_receptor_backbone = True,
    use_esm=False,
    embed_unimol_reprs=False,
    unimol_reprs=None
    ):
    lig_pos = raw_data['ligand'].pos
    lig_len = lig_pos.shape[0]

    lig_atom_positions = torch.zeros((lig_len, 37, 3))
    lig_atom_positions[:, 1] = lig_pos
    lig_atom_mask = torch.zeros((lig_len, 37))
    lig_atom_mask[:, 1] = 1.0
    rec_atom_positions = rec_feature['all_atom_positions']
    rec_atom_mask = rec_feature['all_atom_mask']
    rec_len = rec_atom_positions.shape[0]

    all_atom_positions = torch.cat([rec_atom_positions, lig_atom_positions], dim=0)
    all_atom_mask = torch.cat([rec_atom_mask, lig_atom_mask], dim=0)

    segment = torch.LongTensor([0] * rec_len + [1] * lig_len)
    frame_len = rec_len + lig_len
    rec_single = torch.ones((rec_feature['aatype'].shape[0],1))
    lig_single = make_ligand_node_feature(raw_data['ligand']['x'])
    if embed_unimol_reprs:
        lig_unimol_single = torch.from_numpy(unimol_reprs['atomic_reprs'][0])
        merged_lig_single_unimol = torch.zeros((frame_len, UNIMOL_REPRS_DIM), dtype=torch.float)
        merged_lig_single_unimol[rec_len:] = lig_unimol_single

    rec_single_dim = rec_single.shape[-1]
    lig_single_dim = lig_single.shape[-1]
    single = merged_lig_single = torch.zeros((frame_len, rec_single_dim + lig_single_dim), dtype=torch.float)
    single[:rec_len, :rec_single_dim] = rec_single
    single[rec_len:, rec_single_dim:] = lig_single
    merged_lig_single[rec_len:, rec_single_dim:] = lig_single

    single = torch.cat([single, F.one_hot(segment, num_classes=2)], dim=-1)

    lig_edge = raw_data['ligand', 'lig_bond', 'ligand']
    lig_edge_index = lig_edge.edge_index
    lig_edge_attr = lig_edge.edge_attr
    lig_edge_mask = torch.ones((lig_edge_attr.shape[0],))
    lig_angle_index, lig_torsion_index = make_angle_torsion_index(lig_edge_index+rec_len)
    residx = rec_feature['residx']
    if len(lig_torsion_index)<1:
        return None

    rec_edge_index = torch.stack([torch.arange(rec_len-1), torch.arange(rec_len-1)+1], dim=0)
    st, ed = rec_edge_index[0], rec_edge_index[1]
    is_covalent = (residx[ed] - residx[st]) == 1
    rec_edge_index = rec_edge_index[:, is_covalent]
    rev_edge_index = rec_edge_index.clone()
    rev_edge_index[0] = rec_edge_index[1]
    rev_edge_index[1] = rec_edge_index[0]
    rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
    num_rec_edge = rec_edge_index.shape[1]

    edge_index = torch.cat([rec_edge_index, lig_edge_index + rec_len], dim=1)
    merged_lig_edge_index = torch.cat([torch.ones_like(rec_edge_index) * -2, lig_edge_index + rec_len], dim=1)
    edge_attr = merged_lig_edge_attr = torch.zeros((edge_index.shape[1], lig_edge_attr.shape[1] +1))
    edge_attr[:num_rec_edge, 0] = 1
    edge_attr[num_rec_edge:, 1:] = lig_edge_attr
    merged_lig_edge_attr[num_rec_edge:, 1:] = lig_edge_attr

    residx = torch.cat(
        [residx, torch.arange(lig_len ) + residx.max() + 100], dim=0
    )

    residx_=(residx - torch.min(residx) + 1).tolist()
    residx=torch.tensor(residx_)

    prot_only_resix = rec_feature['residx']
    prot_only_resix_ = (prot_only_resix - torch.min(prot_only_resix) + 1).tolist()
    prot_only_resix=torch.tensor(prot_only_resix_)

    lig_mask = torch.ones((frame_len,))
    lig_mask[:rec_len] = 0

    aatype = torch.cat([ rec_feature['aatype']+4, torch.LongTensor([24]*lig_len)], dim=0)
    node_mask = torch.ones((frame_len,))
    prot_node_mask = torch.ones((rec_len,))
    shuffle_index = np.arange(frame_len)
    inverse_shuffle_index = shuffle_index
    coords = all_atom_positions[:, [0,1,2,4]]

    tokens = aatype
    prot_tokens = rec_feature['aatype']+4
    atom_mask = all_atom_mask[:, [0,1,2,4]]

    confidence = torch.ones((coords.shape[0]))
    feature_dict = {
        'coords': coords,
        'residx': residx,
        'atom_mask': atom_mask,
        'node_mask': node_mask,
        'lig_mask': lig_mask,
        'rec_node_mask': node_mask[:rec_len],
        'lig_node_mask': node_mask[rec_len:],
        'single': single,
        'prot_coords': rec_atom_positions[:, [0,1,2,4]],
        'prot_residx': prot_only_resix,
        'prot_node_mask': prot_node_mask,
        'prot_rec_atom_mask': rec_atom_mask[:, [0,1,2,4]],
        'prot_tokens': prot_tokens,
        'lig_coords': lig_pos,
        'lig_node_attr': merged_lig_single,
        'lig_edge_index': merged_lig_edge_index.transpose(0, 1),
        'lig_edge_attr': merged_lig_edge_attr,
        'edge_index': edge_index.transpose(0, 1),
        'edge_attr': edge_attr,
        'seg_len': torch.LongTensor([rec_len, lig_len]),
        'aatype': aatype,
        'tokens': tokens,
        'target_shuf_index': torch.from_numpy(shuffle_index),
        'target_inv_shuf_index': torch.from_numpy(inverse_shuffle_index),
        'confidence': confidence,
        'rec_len': torch.tensor([rec_len])

    }
    chainidx = torch.zeros((frame_len,)).long()
    chainidx[:rec_len] = rec_feature['chainidx'] + 1
    chainidx = chainidx + 1
    feature_dict['chainidx'] = chainidx
    feature_dict['prot_chainidx'] = chainidx[:rec_len]
    feature_dict['lig_neighbor_seq_mask'] = torch.cat( [ rec_feature['lig_neighbor_seq_mask'], torch.ones((lig_len)).bool() ])
    feature_dict['pdbname'] = rec_feature['pdbname']

    if embed_unimol_reprs:
        feature_dict['unimol_lig_node_attr'] = merged_lig_single_unimol

    pad_angles = torch.stack([torch.ones(lig_len, 4), torch.zeros(lig_len, 4)], -1)
    chi_angles = torch.cat([rec_feature['chi_angles'], pad_angles], 0)
    alt_chi_angles = torch.cat([rec_feature['alt_chi_angles'], pad_angles], 0)
    feature_dict['chi_angles'] = chi_angles_rad = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
    feature_dict['alt_chi_angles'] = alt_chi_angles_rad = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
    feature_dict['chi_mask'] = torch.cat([rec_feature['chi_mask'], torch.zeros(lig_len, 4)], 0)


    return feature_dict


def make_protein_feature(
    rec_feature, raw_data,
    lig_single_dim, lig_pair_dim,
    sde,
    t=None,
    num_aatypes=22,
    fix_receptor_backbone = True,
    use_esm=False,
    randomly_select_res_as_lig=True,
    selected_re_range=[1, 4],
    ca_radius=15.0,
    embed_unimol_reprs=False
    ):

    rec_atom_positions = rec_feature['all_atom_positions']
    rec_atom_mask = rec_feature['all_atom_mask']
    rec_len = rec_atom_positions.shape[0]

    if randomly_select_res_as_lig:
        sel_lig_res_num = np.random.randint(selected_re_range[0], selected_re_range[1])
        cancidate_aatype_res_idx = np.isin(rec_feature['aatype'].numpy(), fake_lig_restypes_id)
        cancidate_res_idx = np.where(np.all(np.stack([cancidate_aatype_res_idx, rec_feature['potential_lig_res'].numpy()], -1), -1) )[0]
        start_res_idx = np.random.choice(cancidate_res_idx, sel_lig_res_num, replace=False)

        if embed_unimol_reprs:
            lig_single, lig_edge_index, lig_edge_attr, lig_pos, lig_unimol_single = \
                make_fake_lig_pos_from_atom37(start_res_idx, rec_atom_positions, rec_atom_mask, rec_feature['aatype'], make_unimol_feature=True)
        else:
            lig_single, lig_edge_index, lig_edge_attr, lig_pos = \
                make_fake_lig_pos_from_atom37(start_res_idx, rec_atom_positions, rec_atom_mask, rec_feature['aatype'], make_unimol_feature=False)

        lig_len = lig_pos.shape[0]

        lig_atom_positions = torch.zeros((lig_len, 37, 3))
        lig_atom_positions[:, 1] = lig_pos
        lig_atom_mask = torch.zeros((lig_len, 37))
        lig_atom_mask[:, 1] = 1.0

        frame_len = rec_len + lig_len
        segment = torch.LongTensor([0] * rec_len + [1] * lig_len)
        lig_single_dim = lig_single.shape[-1]

        all_atom_positions = torch.cat([rec_atom_positions, lig_atom_positions], dim=0)
        all_atom_mask = torch.cat([rec_atom_mask, lig_atom_mask], dim=0)

    else:
        segment = torch.LongTensor([0] * rec_len )
        frame_len = rec_len
        lig_single_dim = 199
        lig_len = 0

        all_atom_positions = rec_atom_positions
        all_atom_mask = rec_atom_mask

    rec_single = torch.ones((rec_feature['aatype'].shape[0],1))
    rec_single_dim = rec_single.shape[-1]

    single = merged_lig_single = torch.zeros((frame_len, rec_single_dim + lig_single_dim), dtype=torch.float)
    single[:rec_len, :rec_single_dim] = rec_single
    if randomly_select_res_as_lig:
        single[rec_len:, rec_single_dim:] = lig_single
        merged_lig_single[rec_len:, rec_single_dim:] = lig_single

    if embed_unimol_reprs:
        merged_lig_single_unimol = torch.zeros((frame_len, UNIMOL_REPRS_DIM), dtype=torch.float)
        merged_lig_single_unimol[rec_len:] = lig_unimol_single

    single = torch.cat([single, F.one_hot(segment, num_classes=2)], dim=-1)
    residx = rec_feature['residx']

    residx = torch.concat([(residx[0]-1)[None], residx,(residx[-1]+1)[None]])
    rec_edge_index = torch.stack([torch.arange(1,  rec_len-2), torch.arange(1, rec_len-2)+1], dim=0)
    st, ed = rec_edge_index[0], rec_edge_index[1]
    is_covalent = (residx[ed] - residx[st]) == 1
    rec_edge_index = rec_edge_index[:, is_covalent]
    rev_edge_index = rec_edge_index.clone()
    rev_edge_index[0] = rec_edge_index[1]
    rev_edge_index[1] = rec_edge_index[0]
    rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
    num_rec_edge = rec_edge_index.shape[1]

    if randomly_select_res_as_lig:
        edge_index = torch.cat([rec_edge_index, lig_edge_index + rec_len], dim=1)
        merged_lig_edge_index = torch.cat([torch.ones_like(rec_edge_index) * -2, lig_edge_index + rec_len], dim=1)
        edge_attr = merged_lig_edge_attr = torch.zeros((edge_index.shape[1], lig_edge_attr.shape[1] +1))

        edge_attr[:num_rec_edge, 0] = 1
        edge_attr[num_rec_edge:, 1:] = lig_edge_attr
        merged_lig_edge_attr[num_rec_edge:, 1:] = lig_edge_attr

        residx = torch.cat(
            [residx, torch.arange(lig_len ) + residx.max() + 100], dim=0
        )
    else:
        edge_index = rec_edge_index
        merged_lig_edge_index = torch.cat([torch.ones_like(rec_edge_index) * -2], dim=1)
        edge_attr = torch.zeros((edge_index.shape[1], 5))
        edge_attr[:num_rec_edge, 0] = 1

    residx_=(residx - torch.min(residx) + 1).tolist()
    residx=torch.tensor(residx_)

    prot_only_resix = rec_feature['residx']
    prot_only_resix_ = (prot_only_resix - torch.min(prot_only_resix) + 1).tolist()
    prot_only_resix=torch.tensor(prot_only_resix_)

    lig_mask = torch.ones((frame_len,))
    lig_mask[:rec_len] = 0

    if randomly_select_res_as_lig:
        aatype = torch.cat([ rec_feature['aatype']+4, torch.LongTensor([24]*lig_len)], dim=0)
        seg_len = torch.LongTensor([rec_len, lig_len])

    else:
        aatype = rec_feature['aatype']+4
        seg_len = torch.LongTensor([rec_len, 0])
    node_mask = torch.ones((frame_len,))
    if randomly_select_res_as_lig:
        node_mask[torch.tensor(start_res_idx.tolist()).long()] = 0.
    prot_node_mask = torch.ones((rec_len,))
    shuffle_index = np.arange(frame_len)
    inverse_shuffle_index = shuffle_index
    # coords = all_atom_positions[:,:3]
    coords = all_atom_positions[:, [0,1,2,4]]

    tokens = aatype
    prot_tokens = rec_feature['aatype']+4
    atom_mask = all_atom_mask[:, [0,1,2,4]]
    confidence = torch.ones((coords.shape[0]))


    feature_dict = {
        'coords': coords,
        'residx': residx,
        'atom_mask': atom_mask,
        'node_mask': node_mask,
        'lig_mask': lig_mask,
        'rec_node_mask': node_mask[:rec_len],
        'lig_node_mask': node_mask[rec_len:],
        'single': single,
        'prot_coords': rec_atom_positions[:, [0,1,2,4]],
        'prot_residx': prot_only_resix,
        'prot_node_mask': prot_node_mask,
        'prot_rec_atom_mask': rec_atom_mask[:, [0,1,2,4]],
        'prot_tokens': prot_tokens,
        'lig_coords': lig_pos,
        'lig_node_attr': merged_lig_single,
        'lig_edge_index': merged_lig_edge_index.transpose(0, 1),
        'lig_edge_attr': merged_lig_edge_attr,
        'edge_index': edge_index.transpose(0, 1),
        'edge_attr': edge_attr,
        'seg_len': seg_len,
        'aatype': aatype,
        'tokens': tokens,
        'target_shuf_index': torch.from_numpy(shuffle_index),
        'target_inv_shuf_index': torch.from_numpy(inverse_shuffle_index),
        'confidence': confidence,
        'rec_len': torch.tensor([rec_len])
    }

    chainidx = torch.zeros((frame_len,)).long()
    chainidx[:rec_len] = rec_feature['chainidx']  + 1
    chainidx = chainidx + 1
    feature_dict['chainidx'] = chainidx

    feature_dict['prot_chainidx'] = chainidx[:rec_len]
    if randomly_select_res_as_lig:
        seq_mask = seq_given_mask(rec_atom_positions[:, 1], lig_pos, ca_radius=ca_radius)
        feature_dict['lig_neighbor_seq_mask'] = torch.cat( [ seq_mask, torch.ones((lig_len)).bool() ])
    else:
        feature_dict['lig_neighbor_seq_mask'] = rec_feature['lig_neighbor_seq_mask']
    feature_dict['pdbname'] = rec_feature['pdbname']

    if embed_unimol_reprs:
        feature_dict['unimol_lig_node_attr'] = merged_lig_single_unimol

    pad_angles = torch.stack([torch.ones(lig_len, 4), torch.zeros(lig_len, 4)], -1)
    chi_angles = torch.cat([rec_feature['chi_angles'], pad_angles], 0)
    alt_chi_angles = torch.cat([rec_feature['alt_chi_angles'], pad_angles], 0)
    feature_dict['chi_angles'] = chi_angles_rad = torch.atan2(chi_angles[..., 0], chi_angles[..., 1])
    feature_dict['alt_chi_angles'] = alt_chi_angles_rad = torch.atan2(alt_chi_angles[..., 0], alt_chi_angles[..., 1])
    feature_dict['chi_mask'] = torch.cat([rec_feature['chi_mask'], torch.zeros(lig_len, 4)], 0)

    return feature_dict


def make_af_feature(
    rec_feature,
    num_aatypes=22,
    fix_receptor_backbone = True,
    use_esm=False,
    ):
    rec_atom_positions = rec_feature['coords']
    atom_mask = ((rec_atom_positions.norm(dim=-1))<10000).float()
    rec_atom_positions = torch.concat([torch.zeros(1,3,3), rec_atom_positions,torch.zeros(1,3,3)])
    rec_atom_mask = torch.concat([torch.zeros(1, 3), atom_mask, torch.zeros(1,3)])
    rec_len = rec_atom_positions.shape[0]
    all_atom_positions = rec_atom_positions
    all_atom_mask = rec_atom_mask


    segment = torch.LongTensor([0] * rec_len )
    frame_len = rec_len
    rec_single = torch.ones((rec_feature['aatype'].shape[0],1))
    rec_single_dim = rec_single.shape[-1]
    lig_single_dim = 199

    single = torch.zeros((frame_len, rec_single_dim + lig_single_dim), dtype=torch.float)
    single[1:rec_len-1, :rec_single_dim] = rec_single
    single = torch.cat([single, F.one_hot(segment, num_classes=2)], dim=-1)
    residx = rec_feature['residx']

    residx = torch.concat([(residx[0]-1)[None], residx,(residx[-1]+1)[None]])
    rec_edge_index = torch.stack([torch.arange(1,  rec_len-2), torch.arange(1, rec_len-2)+1], dim=0)
    st, ed = rec_edge_index[0], rec_edge_index[1]
    is_covalent = (residx[ed] - residx[st]) == 1
    rec_edge_index = rec_edge_index[:, is_covalent]
    rev_edge_index = rec_edge_index.clone()
    rev_edge_index[0] = rec_edge_index[1]
    rev_edge_index[1] = rec_edge_index[0]
    rec_edge_index = torch.cat([rec_edge_index, rev_edge_index], dim=-1)
    num_rec_edge = rec_edge_index.shape[1]
    edge_index = rec_edge_index
    edge_attr = torch.zeros((edge_index.shape[1], 5))
    edge_attr[:num_rec_edge, 0] = 1

    residx_=(residx - torch.min(residx) + 1).tolist()
    residx=torch.tensor(residx_)

    lig_mask = torch.ones((frame_len,))
    lig_mask[:rec_len] = 0

    aatype = torch.cat([torch.tensor(33)[None], rec_feature['aatype']+4, torch.tensor(25)[None], ], dim=0)
    node_mask = torch.ones((frame_len,))
    shuffle_index = np.arange(frame_len)
    inverse_shuffle_index = shuffle_index
    coords = all_atom_positions[:,:3]
    tokens = aatype
    atom_mask = all_atom_mask[:,:3]
    confidence = torch.concat([torch.tensor(0.0)[None], rec_feature['confidence'], torch.tensor(0.0)[None]])
    feature_dict = {
        'coords': coords,
        'residx': residx,
        'atom_mask': atom_mask,
        'node_mask': node_mask,
        'lig_mask': lig_mask,
        'rec_node_mask': node_mask[:rec_len],
        'lig_node_mask': node_mask[rec_len:],
        'single': single,
        'seg_len': torch.LongTensor([rec_len, 0]),
        'aatype': aatype,
        'tokens': tokens,
        'edge_index': edge_index.transpose(0, 1),
        'edge_attr': edge_attr,
        'target_shuf_index': torch.from_numpy(shuffle_index),
        'target_inv_shuf_index': torch.from_numpy(inverse_shuffle_index),
        'confidence': confidence
    }
    chainidx = torch.ones((frame_len,)).long()
    # ch1, ch2 = rec_feature['chainidx'][0], rec_feature['chainidx'][-1]
    # chainidx[:rec_len] = torch.concat([ch1[None], rec_feature['chainidx'],ch2[None] ])  + 1
    chainidx = chainidx + 1
    feature_dict['chainidx'] = chainidx

    return feature_dict
