import torch
import torch.nn.functional as F
import numpy as np

from . import rigid

# def get_quataffine(pos):
#     assert len(pos.shape)
#     nres, natoms, _ = pos.shape
#     assert natoms == 5
#     alanine_idx = residue_constants.restype_order_with_x["A"]
#     aatype = torch.LongTensor([alanine_idx] * nres)
#     all_atom_positions = F.pad(pos, (0,0,0,37-5), "constant", 0)
#     all_atom_mask = torch.ones(nres, 37)
#     frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

#     return frame_dict['rigidgroups_gt_frames']


def compute_chain_center_mass(merged_coords, merged_chain_label, coords_mask=None):
    reduced_chain_label_list = list(set(merged_chain_label))
    center_mass_dict = {}
    for chain_label in reduced_chain_label_list:
        if coords_mask is not None:
            try: 
                chain_coords = merged_coords[coords_mask][np.array(merged_chain_label)[coords_mask] == chain_label]
            except:
                chain_coords = merged_coords[np.array(merged_chain_label) == chain_label]
        else:
            chain_coords = merged_coords[np.array(merged_chain_label) == chain_label]
        ca_chain_coords = chain_coords[:, 1]
        chain_ca_mass_center = ca_chain_coords.mean(0)
        center_mass_dict[chain_label] = chain_ca_mass_center

    uncollate_center_mass = np.stack([center_mass_dict[res_chain_label] for res_chain_label in merged_chain_label])
    return torch.from_numpy(uncollate_center_mass)


def update_rigid_pos_new(pos, translation, rotation):
    assert len(pos.shape) == 3
    L, N, _ = pos.shape
    ca_mass_pos = pos[:, 1].mean(0)
    new_ca_mass_pos = ca_mass_pos + translation
    roted_pos = torch.matmul(pos.reshape(-1, 3) - ca_mass_pos, rotation)
    updated_pos = roted_pos.reshape(L, N, -1)
    updated_pos = updated_pos + new_ca_mass_pos[None, None]
    
    return updated_pos


def add_pseudo_c_beta_from_gly(pos):
    vec_ca = pos[:, 1]
    vec_n = pos[:, 0]
    vec_c = pos[:, 2]
    vec_o = pos[:, 3]
    b = vec_ca - vec_n
    c = vec_c - vec_ca
    a = torch.cross(b, c)
    vec_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
    return torch.stack([vec_n, vec_ca, vec_c, vec_cb, vec_o]).permute(1,0,2)


def permute_between_ss_from_pos(
    gt_pos, # L, 5, 3
    sstype, # L, 2 (ss3, ss8) use ss3 only
    ca_noise_scale, # list: ss translation scale, [5, 8]
    quat_noise_scale, 
    white_noise_scale, # masked residue translation scale, 25
    uncollate_center_mass, # chain mass center, L, 3
    sketch_data, # bool: true using sketch data
    ss_mask_p_range, # list: mask ss prob # [0.4, 0.7]
    loop_mask_p_range # list: mask loop prob # [0.8, 0.9]
    ):
    ss3type = sstype[:, 0]
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
    ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)
    center_mass_for_start_sstypes = torch.index_select(uncollate_center_mass, 0, ss_start_indexs)

    if isinstance(ca_noise_scale, list):
        ca_noise_scale = np.random.uniform(ca_noise_scale[0], ca_noise_scale[1], 1)[0]

    assert isinstance(ss_mask_p_range, list)
    ss_mask_p = np.random.uniform(ss_mask_p_range[0], ss_mask_p_range[1], 1)[0]
    assert isinstance(loop_mask_p_range, list)
    loop_mask_p = np.random.uniform(loop_mask_p_range[0], loop_mask_p_range[1], 1)[0]

    traj_coords = []
    for ss_idx, ss in enumerate(start_sstypes):
        ss_len = ss_lens[ss_idx]
        ss_start_index = ss_start_indexs[ss_idx]
        ss_end_index = ss_end_indexs[ss_idx]
        gt_ss_pos = gt_pos[ss_start_index: ss_end_index+1]

        if ((ss_len > 2) and (ss != 1)):
            if np.random.rand(1)[0] > ss_mask_p:
                ss_frame = rigid.rigid_from_3_points(gt_ss_pos[ss_len.item()//2, 0], gt_ss_pos[ss_len.item()//2, 1], gt_ss_pos[ss_len.item()//2, 2])
                ss_quat = rigid.rot_to_quat(ss_frame['rot'])
                # traj_quat = updated_noising_quat(ss_quat[None], quat_noise_scale)
                qT = rigid.rand_quat(ss_quat.shape[:-1]).to(ss_quat.device)

                new_traj_rot = rigid.quat_to_rot(qT)

                updated_traj_trans = torch.randn(3) * ca_noise_scale

                if sketch_data:
                    sketch_ss_pos = add_pseudo_c_beta_from_gly(
                        torch.from_numpy(
                            gen_peptides_ref_native_peptides(gt_ss_pos.numpy()[:, :4], 
                            SS3_num_to_name[ss.item()])).float())
                    traj_ss_pos = update_rigid_pos_new(sketch_ss_pos, updated_traj_trans, new_traj_rot)
                else:
                    traj_ss_pos = update_rigid_pos_new(gt_ss_pos, updated_traj_trans, new_traj_rot)
                traj_coords.append(traj_ss_pos)
            else:
                noising_quat = rigid.rand_quat([1, ss_len])
                noising_coord = torch.randn(1, ss_len, 3) * white_noise_scale + center_mass_for_start_sstypes[ss_idx][None, None]
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
                traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))

        else:
            if np.random.rand(1)[0] > loop_mask_p:
                traj_coords.append(add_pseudo_c_beta_from_gly(gt_ss_pos))
            else:
                noising_quat = rigid.rand_quat([1, ss_len])
                noising_coord = torch.randn(1, ss_len, 3) * white_noise_scale + center_mass_for_start_sstypes[ss_idx][None, None]
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
                traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))
            
    traj_coords = torch.cat(traj_coords)
    # traj_flat12s = get_quataffine(traj_coords)

    return traj_coords #, traj_flat12s


