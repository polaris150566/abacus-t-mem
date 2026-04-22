import torch
from ..data.utils.bonds import(
    get_ligand_bond_lens,
    get_prot_bond_lens,
    get_ligand_angles_cos,
    get_prot_angles_cos,
    get_prot_torsion_sincos,
    get_ligand_torsions_sincos,
    get_prot_phipsi,
)
from ..modules.gvp_modules.util import get_rotation_frames,rotate
# from unifold.data.data_ops import atom37_to_torsion_angles_new
import sys
# sys.path.append('/train14/superbrain/lili27/protein/alphafold2')
# from alphafold.all_atom import atom37_to_torsion_angles

def get_cent_coord(coords,seg_lens):
    lens = seg_lens[:,1]
    max_len = lens.max()
    indices = torch.arange(max_len).expand(len(lens), max_len).to(lens.device)
    mask = indices < lens.unsqueeze(1)
    indices = indices*mask
    indices_ = indices + seg_lens[:,0:1]
    lig_coords = torch.gather(coords[..., 1, :],1,indices_[...,None].tile(1,1,3))
    mask = mask.type_as(coords)[...,None]
    lig_cent = torch.sum(lig_coords*mask,[1])/(1e-8+torch.sum(mask,[1]))
    return lig_cent

def cent_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']
    seg_lens = batch['seg_len']
    sigma_t = batch['sigma_t']
    pred_cent = get_cent_coord(x_0_, seg_lens)
    ref_cent = get_cent_coord(x_0, seg_lens)
    loss = torch.sum((pred_cent-ref_cent)**2,-1)
    weight_t = 1.0 / (sigma_t[:,0,0,0] ) **2
    loss = loss*weight_t

    return loss
# def angle_diff(a,b):
#     return abs(((a-b)+180)%360-180)
def angle_diff(a,b):
    return torch.abs(((a-b)+torch.pi)%(2*torch.pi)-torch.pi)

    
def length_to_mask(lens):
    max_len = lens.max()
    mask = torch.arange(max_len).expand(len(lens), max_len).to(lens.device)
    mask = mask < lens.unsqueeze(1)
    return mask.float()

def length_to_mask_v2(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    mask = torch.arange(max_len).expand(len(lens), max_len).to(lens.device)
    mask = mask < lens.unsqueeze(1)
    return mask.float()



def score_matching_loss(out, batch, fixbb=False):
    sigma_t = batch['sigma_t']
    x_0 = batch['atom_positions_0']
    atom_mask = batch['atom_mask']
    x_0_ = batch ['x_0_']
    weight_t = 1.0 / (sigma_t[:,0,0,0] ) **2
    mse = torch.sum((x_0_ - x_0) ** 2, dim=-1)
    _, max_len, _ = atom_mask.shape
    rec_mask = length_to_mask_v2(batch['seg_len'][:,0], max_len)
    rec_atom_mask = rec_mask[...,None]*atom_mask
    lig_atom_mask = (1-rec_mask[...,None])*atom_mask

    rec_loss = torch.sum(mse * rec_atom_mask, [1,2]) / (torch.sum(rec_atom_mask, [1, 2]) + 1e-8) * weight_t
    lig_loss = torch.sum(mse * lig_atom_mask, [1,2]) / (torch.sum(lig_atom_mask, [1, 2]) + 1e-8) * weight_t


    return rec_loss

def score_matching_loss(batch):
    sigma_t = batch['sigma_t']
    x_0 = batch['atom_positions_0']
    atom_mask = batch['atom_mask']

    x_0_ = batch ['x_0_']
    weight_t = 2.0* batch['t'][:,None] / ((sigma_t[:,:,0,0] ) **2 + 1e-6)
    mse = torch.sum((x_0_ - x_0) ** 2, dim=-1)* weight_t[...,None]
    _, max_len, _ = atom_mask.shape
    rec_mask = length_to_mask_v2(batch['seg_len'][:,0], max_len)
    rec_atom_mask = rec_mask[...,None]*atom_mask
    lig_atom_mask = (1-rec_mask[...,None])*atom_mask

    rec_loss = torch.sum(mse * rec_atom_mask, [1,2]) / (torch.sum(rec_atom_mask, [1, 2]) + 1e-8) 
    lig_loss = torch.sum(mse * lig_atom_mask, [1,2]) / (torch.sum(lig_atom_mask, [1, 2]) + 1e-8) 
    return rec_loss, lig_loss

def score_matching_loss_lig(batch):
    sigma_t = batch['sigma_t']
    x_0 = batch['atom_positions_0']
    atom_mask = batch['atom_mask']
    x_0_ = batch ['x_0_']
    weight_t = 2.0 / (sigma_t[:,0,0,0] ) **2 * batch['t']
    mse = torch.sum((x_0_ - x_0) ** 2, dim=-1)
    lig_loss = torch.sum(mse * atom_mask, [1,2]) / (torch.sum(atom_mask, [1, 2]) + 1e-8) * weight_t
    return lig_loss


def get_all_dist_from_coord(coord):
    B, N = coord.shape[:2]
    diff = coord[..., None,:]- coord[...,None,:,:]
    dist = diff.norm(dim=-1)
    return dist

def get_sidechain_dist(coord):
    pass

def reciprocal_mse(r1, r2):
    r1 = torch.clamp(r1, 0.01, 1e3)
    r2 = torch.clamp(r2, 0.01, 1e3)
    dist = (1.0/r1 -1.0/r2)**2
    return dist


def reciprocal_loss(batch, ca_thr=9):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']  # [B, N, 37, 3]
    atom_mask =  batch['atom_mask']

    #mainchain
    pred_ca_dist = get_all_dist_from_coord(x_0_[...,1,:])
    ref_ca_dist = get_all_dist_from_coord(x_0[...,1,:])
    mainchain_loss =  reciprocal_mse(pred_ca_dist, ref_ca_dist)

    ref_ca_mask = ref_ca_dist < ca_thr
    N = x_0.shape[1]
    # ref_ca_mask[:,range(N),range(N)] = False
    def get_sidechain_dist(coord):
        ind1, ind2, ind3 = torch.where(ref_ca_mask)
        src = coord[ind1,ind2]
        dest = coord[ind1,ind3]
        mask_src = atom_mask[ind1,ind2]
        mask_dest = atom_mask[ind1,ind3]
        mask_all = mask_src[...,None] * mask_dest[:,None]
        dist = (src[...,None,:]-dest[...,None,:,:]).norm(dim=-1)
        dist_preserve = dist[mask_all.bool()]
        # import pdb; pdb.set_trace()
        return dist_preserve
    pred_sidechain_dist = get_sidechain_dist(x_0_)
    ref_sidechain_dist = get_sidechain_dist(x_0)

    sidechain_loss = reciprocal_mse (pred_sidechain_dist, ref_sidechain_dist)
    loss = sidechain_loss.mean().unsqueeze(0)
    return loss

def ligand_bond_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_bond_len = get_ligand_bond_lens(x_0_, batch['lig_edge_index'])
    ref_bond_len = get_ligand_bond_lens(x_0, batch['lig_edge_index'])
    bond_error = (pred_bond_len - ref_bond_len) ** 2
    bond_error = torch.clamp(bond_error, 0, 40)
    lig_edge_len = batch['edge_size'][..., 1]
    lig_edge_mask = length_to_mask(lig_edge_len)
    loss = torch.sum(lig_edge_mask * bond_error, dim=-1) / (torch.sum(lig_edge_mask, dim=-1) + 1e-8)
    return loss

def ligand_angle_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_cos = get_ligand_angles_cos(x_0_, batch['lig_angle_index'])
    ref_cos = get_ligand_angles_cos(x_0, batch['lig_angle_index'])

    error = (pred_cos - ref_cos)**2
    mask = length_to_mask(batch['lig_angle_size'].squeeze(-1))
    loss = torch.sum(error * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-8)
    return loss

def ligand_torsion_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_sincos = get_ligand_torsions_sincos(x_0_, batch['lig_torsion_index'])
    ref_sincos = get_ligand_torsions_sincos(x_0, batch['lig_torsion_index'])
    error = (pred_sincos - ref_sincos) ** 2
    error = torch.sum(error, -1)
    mask = length_to_mask(batch['lig_torsion_size'].squeeze(-1))
    loss = torch.sum(error * mask, dim=-1) / (torch.sum(mask, dim=-1) + 1e-8)
    return loss

def protein_bond_loss(batch):
    # import pdb; pdb.set_trace()
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']
    pred_len, bond_mask = get_prot_bond_lens(x_0_, batch['aatype'])
    ref_len, _ = get_prot_bond_lens(x_0, batch['aatype'])

    rec_len = batch['seg_len'][..., 0]
    rec_mask = length_to_mask(rec_len)

    max_rec_len = rec_mask.shape[1]
    pred_len = pred_len[:, :max_rec_len]
    ref_len = ref_len[:, :max_rec_len]
    bond_mask = bond_mask[:, :max_rec_len]
    bond_mask = bond_mask * rec_mask[..., None]

    error = (pred_len - ref_len) ** 2
    loss = torch.sum(error * bond_mask, [1, 2]) / (torch.sum(bond_mask, [1, 2]) + 1e-8)
    return loss

def protein_angle_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_angle_cos, angle_mask = get_prot_angles_cos(x_0_, batch['aatype'])
    ref_angle_cos, _ = get_prot_angles_cos(x_0, batch['aatype'])

    rec_len = batch['seg_len'][..., 0]
    rec_mask = length_to_mask(rec_len)

    max_rec_len = rec_mask.shape[1]
    pred_angle_cos = pred_angle_cos[:, :max_rec_len]
    ref_angle_cos = ref_angle_cos[:, :max_rec_len]
    angle_mask = angle_mask[:, :max_rec_len]
    angle_mask = angle_mask * rec_mask[..., None]

    error = (pred_angle_cos - ref_angle_cos) ** 2
    loss = torch.sum(error * angle_mask, [1, 2]) / (torch.sum(angle_mask, [1, 2]) + 1e-8)
    return loss


def protein_torsion_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_torsion_sincos, torsion_mask = get_prot_torsion_sincos(x_0_, batch['aatype'])
    ref_torsion_sincos, _ = get_prot_torsion_sincos(x_0, batch['aatype'])

    rec_len = batch['seg_len'][..., 0]
    rec_mask = length_to_mask(rec_len)

    max_rec_len = rec_mask.shape[1]
    pred_torsion_sincos = pred_torsion_sincos[:, :max_rec_len]
    ref_torsion_sincos = ref_torsion_sincos[:, :max_rec_len]
    torsion_mask = torsion_mask[:, :max_rec_len]
    torsion_mask = torsion_mask * rec_mask[..., None]

    error = (pred_torsion_sincos - ref_torsion_sincos) ** 2
    # import pdb; pdb.set_trace()
    loss = torch.sum(error * torsion_mask[..., None], [1, 2, 3]) / (torch.sum(torsion_mask, [1, 2]) + 1e-8)
    return loss

def torsion_error(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    def get_torsion(sincos):
        torsion = torch.atan2(sincos[...,0],sincos[...,1])
        return torsion

    def mean_absolute_error(ref_k, pred_k, mask):
        # diff = torch.abs(ref_k -pred_k)
        diff =  angle_diff(ref_k, pred_k)
        error = torch.sum(diff*mask, [0])/(1e-8+torch.sum(mask,[0]))*(180/torch.pi)
        return error,torch.sum(mask,0)
        
    pred_protein = {'all_atom_positions':x_0_,"aatype":batch['aatype'],"all_atom_mask":batch['all_atom_mask']}
    ref_protein = {'all_atom_positions':x_0,"aatype":batch['aatype'],"all_atom_mask":batch['all_atom_mask']}

    pred_torsion = atom37_to_torsion_angles_new(pred_protein)
    pred_torsion_sincos, torsion_mask = pred_torsion['torsion_angles_sin_cos'],pred_torsion["torsion_angles_mask"]
    # pred_torsion_sincos = atom37_to_torsion_angles_new(pred_protein)['torsion_angles_sin_cos']
    ref_torsion_sincos = atom37_to_torsion_angles_new(ref_protein)['torsion_angles_sin_cos']
    pred_angle= get_torsion(pred_torsion_sincos)[0]
    ref_angle = get_torsion(ref_torsion_sincos)[0]
    ref_k4 = ref_angle[...,3:]
    pred_k4 = pred_angle[...,3:]
    torsion_mask = torsion_mask[0,:,3:]
    error, len_ = mean_absolute_error(ref_k4, pred_k4, torsion_mask)

    return error, len_


def protein_phipsi_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    pred_phipsi = get_prot_phipsi(x_0_)
    ref_phipsi = get_prot_phipsi(x_0)

    rec_len = batch['seg_len'][..., 0] - 1
    rec_mask = length_to_mask(rec_len)

    max_rec_len = rec_mask.shape[1]
    pred_phipsi = pred_phipsi[:, :max_rec_len]
    ref_phipsi = ref_phipsi[:, :max_rec_len]
    residx = batch['residx'][:, :max_rec_len+1]
    phipsi_mask = (residx[:, 1:] - residx[:, :-1] == 1).float() * rec_mask

    error = (pred_phipsi - ref_phipsi) ** 2
    loss = torch.sum(error * phipsi_mask[..., None], [1, 2]) / (torch.sum(phipsi_mask, [1]) + 1e-8)
    return loss


def global_to_local_coord(x, frame, mask):
    R, t = frame
    N = x.shape[1]
    x = x[:, None].tile(1, N, 1, 1, 1)
    R = R[:, :, None].tile(1, 1, N, 1, 1)
    t = t[:, :, None, None].tile(1, 1, N, 1, 1)
    # t = t[:, None,:,None].tile(1, N, 1, 1, 1)
    mask = mask[:, None].tile(1, N, 1, 1)
   
    x_l = torch.einsum('...md,...kd->...kd', R.transpose(-1, -2), x - t)
    x_l = x_l * mask[..., None]
    return x_l


def protein_sidechain_loss(batch):
    # import pdb; pdb.set_trace()
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    atom_mask = batch['atom_mask']
    pred_frame = (get_rotation_frames(x_0_), x_0_[..., 1, :])
    ref_frame = (get_rotation_frames(x_0), x_0[..., 1, :])

    pred_x_local = global_to_local_coord(x_0_, pred_frame, atom_mask)
    ref_x_local = global_to_local_coord(x_0, ref_frame, atom_mask)
    # import pdb; pdb.set_trace()
    dist = (pred_x_local - ref_x_local) ** 2
    dist = torch.clamp(dist, 0, 50)

    rec_len = batch['seg_len'][..., 0]
    rec_mask = length_to_mask(rec_len)
    max_rec_len = rec_mask.shape[1]
    seq_mask = torch.zeros(*atom_mask.shape[:2]).to(x_0.device)
    seq_mask[:,:max_rec_len] = rec_mask
    atom_mask = atom_mask.unsqueeze(1)*atom_mask.unsqueeze(2)
    atom_mask [~seq_mask.bool()] = 0
    loss = torch.sum(dist * atom_mask[..., None], [1,2,3,4]) / (torch.sum(atom_mask, [1,2,3]) + 1e-8)
    return loss

def get_hbond_len(coords,index,atom_type):
    src, dest = index[...,0],index[...,1]
    src_type, dest_type = atom_type[...,0],atom_type[...,1]

    x_src = torch.gather(coords, 1, src[...,None, None].tile(1, 1, 37, 3))
    x_dest = torch.gather(coords, 1, dest[...,None, None].tile(1, 1, 37, 3))
    x_src_ = torch.gather(x_src, 2, src_type[...,None, None].tile(1, 1, 1, 3))
    x_dest_ = torch.gather(x_dest, 2, dest_type[...,None, None].tile(1, 1, 1, 3))
    dist = (x_src_-x_dest_).norm(dim=-1)

    return dist


def hbond_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']

    hbond_index = batch['hbond_index']
    hbond_atype = batch['hbond_type']
    hbond_mask = batch['hbond_mask']
    ref_hbond_len = get_hbond_len(x_0,hbond_index,hbond_atype)
    pred_hbond_len = get_hbond_len(x_0_,hbond_index,hbond_atype)

    error = reciprocal_mse(ref_hbond_len,pred_hbond_len).squeeze(-1)
    loss = torch.sum(error * hbond_mask, [1]) / ( torch.sum(hbond_mask,[1]) + 1e-8)
    return loss


def k1k2_loss(batch):
    x_0_ = batch['x_0_']
    x_0 = batch['atom_positions_0']
    def get_angle(sincos):
        # tan = sincos[...,0]/ (1e-8 + sincos[...,1])
        # arctan = torch.arctan(tan)*(180/torch.pi)
        arccos = torch.arccos(sincos[...,1])
        angle_add = torch.zeros_like(sincos[...,0])
        mask = sincos[...,0]<0
        angle_add[mask] = torch.pi
        arccos = arccos + angle_add
        return arccos

    pred_protein = {'all_atom_positions':x_0_,"aatype":batch['aatype'],"all_atom_mask":batch['atom_mask']}
    ref_protein = {'all_atom_positions':x_0,"aatype":batch['aatype'],"all_atom_mask":batch['atom_mask']}

    pred_torsion_sincos = atom37_to_torsion_angles_new(pred_protein)['torsion_angles_sin_cos']
    ref_torsion_sincos = atom37_to_torsion_angles_new(ref_protein)['torsion_angles_sin_cos']
    # pred_torsion_sincos = atom37_to_torsion_angles(batch['aatype'],x_0_, batch['atom_mask'])['torsion_angles_sin_cos']
    # ref_torsion_sincos = atom37_to_torsion_angles(batch['aatype'],x_0, batch['atom_mask'])['torsion_angles_sin_cos']

    pred_angle = pred_torsion_sincos
    ref_angle = ref_torsion_sincos
    
    ref_k1k2 = ref_angle[...,3:5,:]
    pred_k1k2 = pred_angle[...,3:5,:]
    # dist = angle_diff(pred_k1k2,ref_k1k2)
    dist = torch.sum( (pred_k1k2 - ref_k1k2)**2 ,[-2,-1])
    loss = dist.mean([1])
    weight_t = 1.0/ batch['t'].clone()
    weight_t[batch['t']>0.5] = 0
    loss = loss*weight_t
    return loss
