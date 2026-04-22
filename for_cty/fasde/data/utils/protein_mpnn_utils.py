import os
import csv
from dateutil import parser
import torch
import random
import numpy as np

from unicore.utils import (
    batched_gather,
)

from unifold.data.residue_constants import (
    restype_atom37_to_atom14, 
    restype_order_with_x,
    restype_atom37_mask,
)

restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int64,
    )
restype_atom37_mask = torch.tensor(
        restype_atom37_mask, dtype=torch.float32
    )

def sequence_to_aatype(seq, device='cpu'):
    aatype = [restype_order_with_x[aa] for aa in seq]
    return torch.LongTensor(aatype).to(device)

def convert_atom14_to_atom37(data):
    aatype = data['aatype']
    indices = restype_atom37_to_atom14[aatype]
    gt_mask = restype_atom37_mask[aatype]

    all_atom_positions = batched_gather(
        data['all_atom_positions'],
        indices,
        dim=-2,
        num_batch_dims=len(data['all_atom_positions'].shape[:-2]),
    )
    all_atom_positions = all_atom_positions * gt_mask[..., None]

    all_atom_mask = batched_gather(
        data['all_atom_mask'],
        indices,
        dim=-1,
        num_batch_dims=len(data['all_atom_mask'].shape[:-1]),
    ) 
    all_atom_mask = all_atom_mask * gt_mask
    data['aatype'] = aatype
    data['all_atom_positions'] = torch.nan_to_num(all_atom_positions) 
    data['all_atom_mask'] = all_atom_mask

    return data

def loader_pdb(item, data_dir, homo_thr=0.7):
    pdbid,chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s"%(data_dir,pdbid[1:3],pdbid)
    print(PREFIX)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])
        return [convert_atom14_to_atom37(
                    {'seq'    : chain['seq'],
                     'aatype' : sequence_to_aatype(chain['seq']),
                     'all_atom_positions'    : chain['xyz'],
                     'all_atom_mask'   : chain['mask'],
                     'residx' : torch.arange(L).int(),
                     'chidx'  : torch.zeros(L).int(),})]
                # 'masked' : torch.Tensor([0]).int(),
                # 'label'  : item[0]}

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d'%k]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1&s2

        # transform selected chains 
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                mask = chains[c]['mask']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):[xyz_i, mask] for i,xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j> homo_thr])
    # stack all chains in the assembly together
    # seq,xyz,idx,masked = "",[],[],[]
    # seq_list = []
    assem = []
    assem_keys = [k for k in asmb.keys()]
    random.shuffle(assem_keys)
    for counter, k in enumerate(assem_keys):
    # for counter,(k,v) in enumerate(asmb.items()):
        v = asmb[k]
        xyz, mask = v
        L = xyz.shape[0]
        monomer = {
            'seq': chains[k[0]]['seq'],
            'aatype': sequence_to_aatype(chains[k[0]]['seq']),
            'all_atom_positions': xyz,
            'all_atom_mask': mask,
            'residx' : torch.arange(L).int(),
            'chidx'  : torch.zeros(L).int() + counter,}
        assem.append(convert_atom14_to_atom37(monomer))

    #     ###################
    #     seq += chains[k[0]]['seq']
    #     seq_list.append(chains[k[0]]['seq'])
    #     xyz.append(v)
    #     idx.append(torch.full((v.shape[0],),counter))
    #     if k[0] in homo:
    #         masked.append(counter)

    # return {'seq'    : seq,
    #         'xyz'    : torch.cat(xyz,dim=0),
    #         'idx'    : torch.cat(idx,dim=0),
    #         'masked' : torch.Tensor(masked).int(),
    #         'label'  : item[0]}
    return assem


def shuffle_chidx(asmb, residx_gap=100):
    asmb_size = len(asmb)
    if asmb_size == 1:
        return asmb
    
    shuffle_index = np.random.permutation(asmb_size)
    residx_start = 0
    new_asmb = []
    for count, i in enumerate(shuffle_index):
        L = len(asmb[i]['seq'])
        mono = {
            'seq': asmb[i]['seq'],
            'aatype': asmb[i]['aatype'],
            'all_atom_positions': asmb[i]['all_atom_positions'].float(),
            'all_atom_mask': asmb[i]['all_atom_mask'].float(),
            'residx': torch.arange(L).long() + residx_start,
            'chidx': torch.zeros(L).long() + count,
        }

        residx_start += L + residx_gap
        new_asmb.append(mono)
    
    return new_asmb


def concat_assembly(asmb):
    return {
        'seq': ''.join([m['seq'] for m in asmb]),
        'aatype': torch.cat([m['aatype'] for m in asmb], dim=0),
        'all_atom_positions': torch.cat([m['all_atom_positions'] for m in asmb], dim=0),
        'all_atom_mask': torch.cat([m['all_atom_mask'] for m in asmb], dim=0),
        'residx': torch.cat([m['residx'] for m in asmb], dim=0),
        'chidx': torch.cat([m['chidx'] for m in asmb], dim=0),
    }


def crop_assembly(asmb):
    return


def build_training_clusters(params, debug=False):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test
    