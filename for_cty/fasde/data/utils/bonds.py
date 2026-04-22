import torch
import torch.nn.functional as F
import numpy as np

from . import residue_constants as rc

from ...modules.gvp_modules.util import nan_to_num

def make_angle_torsion_index(edge_index):
    edge_index = edge_index.T

    neighbours = {}
    all_nodes = []
    for st, ed in edge_index.numpy():
        if st not in all_nodes:
            all_nodes.append(st)
        if ed not in all_nodes:
            all_nodes.append(ed)

        if st not in neighbours:
            neighbours[st] = []
        if ed not in neighbours:
            neighbours[ed] = []
        if ed not in neighbours[st]:
            neighbours[st].append(ed)
        if st not in neighbours[ed]:
            neighbours[ed].append(st)

    # tor_edge_index = edge_index[edge_mask]
    # tor_edge_mask = torch.ones((tor_edge_index.shape[0])).float()

    angle_index = []
    for nd in all_nodes:
        for prev in neighbours[nd]:
            for next in neighbours[nd]:
                if prev == next:
                    continue

                if [prev, nd, next] in angle_index or [next, nd, prev] in angle_index:
                    continue
                angle_index.append([prev, nd, next])

    # prev_st, next_ed = [], []
    torsion_node_index = []
    for st, ed in edge_index.numpy():
        for prev in neighbours[st]:
            for next in neighbours[ed]:
                if prev in [next, ed] or next in [st, prev]:
                    continue
                
                if [prev, st, ed, next] in torsion_node_index or [next, ed, st, prev] in torsion_node_index:
                    continue
                torsion_node_index.append([prev, st, ed, next])
    
    angle_index = torch.LongTensor(angle_index)
    torsion_node_index = torch.LongTensor(torsion_node_index)
    return angle_index, torsion_node_index
    

residue_bonds = {
    'ALA': 'N-CA,CA-C,C-O,CA-CB', 
    'ARG': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD,CD-NE,NE-CZ,CZ-NH1,CZ-NH2', 
    'ASP': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-OD1,CG-OD2', 
    'ASN': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-OD1,CG-ND2', 
    'CYS': 'N-CA,CA-C,C-O,CA-CB,CB-SG', 
    'GLU': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD,CD-OE1,CD-OE2', 
    'GLN': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD,CD-NE2,CD-OE1', 
    'GLY': 'N-CA,CA-C,C-O', 
    'HIS': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD2,CD2-NE2,NE2-CE1,CE1-ND1,ND1-CG', 
    'ILE': 'N-CA,CA-C,C-O,CA-CB,CB-CG2,CB-CG1,CG1-CD1', 
    'LEU': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD1,CG-CD2', 
    'LYS': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD,CD-CE,CE-NZ', 
    'MET': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-SD,SD-CE', 
    'PHE': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD2,CD2-CE2,CE2-CZ,CZ-CE1,CE1-CD1,CD1-CG', 
    'PRO': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD,CD-N', 
    'SER': 'N-CA,CA-C,C-O,CA-CB,CB-OG', 
    'THR': 'N-CA,CA-C,C-O,CA-CB,CB-CG2,CB-OG1', 
    'TRP': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD1,CD1-NE1,NE1-CE2,CE2-CD2,CD2-CG,CD2-CE3,CE3-CZ3,CZ3-CH2,CH2-CZ2,CZ2-CE2', 
    'TYR': 'N-CA,CA-C,C-O,CA-CB,CB-CG,CG-CD1,CD1-CE1,CE1-CZ,CZ-CE2,CE2-CD2,CD2-CG,CZ-OH', 
    'VAL': 'N-CA,CA-C,C-O,CA-CB,CB-CG1,CB-CG2', 
}

residue_bond_index = {}
for res in residue_bonds.keys():
    bond_index = []
    for b in residue_bonds[res].split(','):
        atom1, atom2 = b.split('-')
        idx1 = rc.atom_types.index(atom1)
        idx2 = rc.atom_types.index(atom2)
        bond_index.append([idx1, idx2])
    bond_index = torch.LongTensor(bond_index)
    angle_index, torsion_index = make_angle_torsion_index(bond_index.T)

    residue_bond_index[res] = {
        'bond_index': bond_index,
        'angle_index': angle_index,
        'torsion_index': torsion_index
    }

max_bonds = max([residue_bond_index[res]['bond_index'].shape[0] for res in residue_bond_index.keys()])
max_angles = max([residue_bond_index[res]['angle_index'].shape[0] for res in residue_bond_index.keys()])
max_torsions = max([residue_bond_index[res]['torsion_index'].shape[0] for res in residue_bond_index.keys()])

AATYPE_BONDS = torch.zeros((20, max_bonds, 2), dtype=torch.long)
AATYPE_ANGLES = torch.zeros((20, max_angles, 3), dtype=torch.long)
AATYPE_TORSIONS = torch.zeros((20, max_torsions, 4), dtype=torch.long)

AATYPE_BONDS_MASK = torch.zeros((20, max_bonds), dtype=torch.float)
AATYPE_ANGLES_MASK = torch.zeros((20, max_angles), dtype=torch.float)
AATYPE_TORSIONS_MASK = torch.zeros((20, max_torsions), dtype=torch.float)

for aatype, aa in enumerate(rc.restypes):
    res = rc.restype_1to3[aa]
    if res not in residue_bond_index:
        continue

    bond_index = residue_bond_index[res]['bond_index']
    mask_bond = (residue_bond_index[res]['bond_index']<3) | (residue_bond_index[res]['bond_index']==4 )
    mask = mask_bond.sum(-1)!=2
    bond_index = bond_index[mask]
    if len(bond_index)>0:
        AATYPE_BONDS[aatype, :bond_index.shape[0]] = bond_index
        AATYPE_BONDS_MASK[aatype, :bond_index.shape[0]] = 1.0

    angle_index = residue_bond_index[res]['angle_index']
    mask_angle = (residue_bond_index[res]['angle_index']<3) | (residue_bond_index[res]['angle_index']==4 )
    mask = mask_angle.sum(-1)!=3
    angle_index = angle_index[mask]
    if len(angle_index)>0:
        AATYPE_ANGLES[aatype, :angle_index.shape[0]] = angle_index
        AATYPE_ANGLES_MASK[aatype, :angle_index.shape[0]] = 1.0

    torsion_index = residue_bond_index[res]['torsion_index']
    mask_torsion = (residue_bond_index[res]['torsion_index']<3) | (residue_bond_index[res]['torsion_index']==4 )
    mask = mask_torsion.sum(-1)!=4
    torsion_index = torsion_index[mask]
    if len(torsion_index)>0:
        AATYPE_TORSIONS[aatype, :torsion_index.shape[0]] = torsion_index
        AATYPE_TORSIONS_MASK[aatype, :torsion_index.shape[0]] = 1.0


def norm(vec, dim=-1, keepdims=False, eps = 1e-10):
    return torch.sqrt(torch.sum(torch.square(vec), dim=dim, keepdims=keepdims) + eps)

def get_prot_bond_lens(coords, aatype):
    device = coords.device
    bond_type_index = aatype.clamp(max=19)
    aatype_bonds = AATYPE_BONDS.to(device)
    aatype_bonds_mask = AATYPE_BONDS_MASK.to(device)

    bonds = aatype_bonds[bond_type_index]
    bond_mask = aatype_bonds_mask[bond_type_index]

    src, dest = bonds[..., 0], bonds[..., 1]
    x_src = torch.gather(coords, 2, src[..., None].tile(1, 1, 1, 3))
    x_dest = torch.gather(coords, 2, dest[..., None].tile(1, 1, 1, 3))
    bond_vec = x_src - x_dest
    bond_len = norm(bond_vec)
    return bond_len, bond_mask

def get_ligand_bond_lens(coords, bond_index):
    lig_coords = coords[..., 1, :]
    src, dest = bond_index[..., 0], bond_index[..., 1]

    x_src = torch.gather(lig_coords, 1, src[..., None].tile(1, 1, 3))
    x_dest = torch.gather(lig_coords, 1, dest[..., None].tile(1, 1, 3))
    bond_vec = x_src - x_dest

    bond_len = norm(bond_vec)
    return bond_len


def cosine_similarity(v1, v2):
    return torch.sum(v1*v2, -1) / (norm(v1) * norm(v2))


def get_prot_angles_cos(coords, aatype):
    device = coords.device
    bond_type_index = aatype.clamp(max=19)
    aatype_bonds = AATYPE_ANGLES.to(device)
    aatype_bonds_mask = AATYPE_ANGLES_MASK.to(device)

    bonds = aatype_bonds[bond_type_index]
    bond_mask = aatype_bonds_mask[bond_type_index]

    prev, cur, next = bonds[..., 0], bonds[..., 1], bonds[..., 2]

    x_prev = torch.gather(coords, 1, prev[..., None].tile(1, 1, 1, 3))
    x_cur = torch.gather(coords, 1, cur[..., None].tile(1, 1, 1, 3))
    x_next = torch.gather(coords, 1, next[..., None].tile(1, 1, 1, 3))
    v1 = x_prev - x_cur
    v2 = x_next - x_cur

    angle_cos = cosine_similarity(v1, v2)
    return angle_cos, bond_mask


def get_ligand_angles_cos(coords, bond_index):
    lig_coords = coords[..., 1, :]

    prev, cur, next = bond_index[..., 0], bond_index[..., 1], bond_index[..., 2]

    x_prev = torch.gather(lig_coords, 1, prev[..., None].tile(1, 1, 3))
    x_cur = torch.gather(lig_coords, 1, cur[..., None].tile(1, 1, 3))
    x_next = torch.gather(lig_coords, 1, next[..., None].tile(1, 1, 3))
    v1 = x_prev - x_cur
    v2 = x_next - x_cur

    angle_cos = cosine_similarity(v1, v2)
    return angle_cos


def torsion(x1, x2, x3, x4):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / norm(b1, keepdims=True) #(torch.linalg.norm(b1, dim=-1, keepdims=True) + 1e-8)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=-1, keepdims=True) * b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*(w + 1e-8), axis=-1)
    # b1xv = torch.cross(b1, v, axisa=2, axisb=2)c
    b1xv = torch.cross(b1, v, dim=-1)
    y = torch.sum(b1xv*w, dim=-1)

    x = nan_to_num(x)
    y = nan_to_num(y)

    xy_norm = torch.sqrt(x**2 + y**2 + 1e-8)
    sin_ = y / xy_norm
    cos_ = x / xy_norm

    sin_cos = torch.stack([sin_, cos_], dim=-1)
    return sin_cos


def get_ligand_torsions_sincos(coords, bond_index):
    lig_coords = coords[..., 1, :]
    prev, src, dest, next = bond_index[..., 0], bond_index[..., 1], bond_index[..., 2], bond_index[..., 3]

    x_prev = torch.gather(lig_coords, 1, prev[..., None].tile(1, 1, 3))
    x_src = torch.gather(lig_coords, 1, src[..., None].tile(1, 1, 3))
    x_dest = torch.gather(lig_coords, 1, dest[..., None].tile(1, 1, 3))
    x_next = torch.gather(lig_coords, 1, next[..., None].tile(1, 1, 3))
    
    torsion_sin_cos = torsion(x_prev, x_src, x_dest, x_next)
    return torsion_sin_cos


def get_prot_torsion_sincos(coords, aatype):
    device = coords.device
    bond_type_index = aatype.clamp(max=19)
    aatype_bonds = AATYPE_TORSIONS.to(device)
    aatype_bonds_mask = AATYPE_TORSIONS_MASK.to(device)

    bonds = aatype_bonds[bond_type_index]
    bond_mask = aatype_bonds_mask[bond_type_index]

    prev, src, dest, next = bonds[..., 0], bonds[..., 1], bonds[..., 2],  bonds[..., 3]

    x_prev = torch.gather(coords, 1, prev[..., None].tile(1, 1, 1, 3))
    x_src = torch.gather(coords, 1, src[..., None].tile(1, 1, 1, 3))
    x_dest = torch.gather(coords, 1, dest[..., None].tile(1, 1, 1, 3))
    x_next = torch.gather(coords, 1, next[..., None].tile(1, 1, 1, 3))
    torsion_sin_cos = torsion(x_prev, x_src, x_dest, x_next)
    return torsion_sin_cos, bond_mask


def get_prot_phipsi(coords):
    n = coords[:, :, 0]
    ca = coords[:, :, 1]
    c = coords[:, :, 2]

    phi_sincos = torsion(c[:, :-1], n[:, 1:], ca[:, 1:], c[:, 1:])
    psi_sincos = torsion(n[:, :-1], ca[:, :-1], c[:, :-1], n[:, 1:])

    return torch.cat([phi_sincos, psi_sincos], dim=-1)
