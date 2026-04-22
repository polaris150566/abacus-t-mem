import time
import torch
import math
import torch.nn as nn
import sys
import numpy as np

from .pifold_utils import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
from .pifold_module import *
from transformers import AutoTokenizer
from .lig_encoder_utils import flatten_graph

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


class PiFold_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(PiFold_Model, self).__init__()
        self.args = args
        self.augment_eps = args.augment_eps
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        self.dihedral_type = args.dihedral_type
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        alphabet = [one for one in 'ACDEFGHIKLMNPQRSTVWYX']
        # self.token_mask = torch.tensor([(one in alphabet) for one in self.tokenizer._token_to_id.keys()])
        
        # self.full_atom_dis = args.full_atom_dis
        
        # node_in = 12

        # node_in = node_in + 9 + 576 # node_in + 9 + 576
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0       ,  0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]

        # prior_matrix = torch.rand(self.args.virtual_num,3)
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])
        # num_va = self.virtual_atoms.shape[0]
        # edge_in = (15 + 9 * num_va + (num_va - 1) * num_va) * 16 + 16 + 7 

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            node_in += pair_num*self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9
        
        edge_in = 0
        if self.args.edge_dist:
            pair_num = 0
            if self.args.Ca_Ca:
                pair_num += 1
            if self.args.Ca_C:
                pair_num += 2
            if self.args.Ca_N:
                pair_num += 2
            if self.args.Ca_O:
                pair_num += 2
            if self.args.C_C:
                pair_num += 1
            if self.args.C_N:
                pair_num += 2
            if self.args.C_O:
                pair_num += 2
            if self.args.N_N:
                pair_num += 1
            if self.args.N_O:
                pair_num += 2
            if self.args.O_O:
                pair_num += 1

            
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            edge_in += pair_num*self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12
        
        if self.args.use_gvp_feat:
            node_in = 12
            edge_in = 48-16
        
        edge_in += 16+16 # position encoding, chain encoding
        # import pdb; pdb.set_trace()

        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, args.updating_edges, args.att_output_mlp, args.node_output_mlp, args.node_net, args.edge_net, args.node_context, args.edge_context)

        # self.decoder = MLPDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu, vocab=len(self.tokenizer._token_to_id))

        self._init_params()

        self.encode_t = 0
        self.decode_t = 0
    
    def forward(self, h_V, h_P, P_idx, batch_id, S=None, AT_test = False, mask_bw = None, mask_fw = None, decoding_order= None):
        # t1 = time.time()
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)

        return h_V
        # t2 = time.time()
        
        # log_probs, logits = self.decoder(h_V, batch_id, self.token_mask)
                
        # # log_probs, logits = self.decoder2(h_V, logits, batch_id)
        # t3 = time.time()

        # self.encode_t += t2-t1
        # self.decode_t += t3-t2
        # # return log_probs, log_probs0
        # return log_probs
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

        # mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        # dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        # D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        # D_max, _ = torch.max(D, -1, keepdim=True)
        # D_adjust = D + (1. - mask_2D) * D_max
        # sampled_top_k = top_k
        # D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        # return D_neighbors, E_idx

    def _get_features(self, X, mask, residue_idx, chain_encoding):
        device = X.device
        # import pdb; pdb.set_trace()
        mask_bool = (mask==1)
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # sequence
        # S = torch.masked_select(S, mask_bool)
        # if score is not None:
        #     score = torch.masked_select(score, mask_bool)
        # chain_mask = torch.masked_select(chain_mask, mask_bool)
        # chain_encoding = torch.masked_select(chain_encoding, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, self.dihedral_type) 
        # V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        # V_direct = node_mask_select(V_direct)
        # E_direct = edge_mask_select(E_direct)
        # E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:,:,0,:]
        atom_Ca = X[:,:,1,:]
        atom_C = X[:,:,2,:]
        atom_O = X[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze())
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze())
                    node_dist.append(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze())
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()
        
        pair_lst = []
        if self.args.Ca_Ca:
            pair_lst.append('Ca-Ca')
        if self.args.Ca_C:
            pair_lst.append('Ca-C')
            pair_lst.append('C-Ca')
        if self.args.Ca_N:
            pair_lst.append('Ca-N')
            pair_lst.append('N-Ca')
        if self.args.Ca_O:
            pair_lst.append('Ca-O')
            pair_lst.append('O-Ca')
        if self.args.C_C:
            pair_lst.append('C-C')
        if self.args.C_N:
            pair_lst.append('C-N')
            pair_lst.append('N-C')
        if self.args.C_O:
            pair_lst.append('C-O')
            pair_lst.append('O-C')
        if self.args.N_N:
            pair_lst.append('N-N')
        if self.args.N_O:
            pair_lst.append('N-O')
            pair_lst.append('O-N')
        if self.args.O_O:
            pair_lst.append('O-O')

        
        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(rbf)

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf))

                for j in range(0, i):
                    edge_dist.append(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf))
                    edge_dist.append(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.args.node_dist:
            h_V.append(V_dist)
        if self.args.node_angle:
            h_V.append(V_angles)
        if self.args.node_direct:
            h_V.append(V_direct)
        
        h_E = []
        if self.args.edge_dist:
            h_E.append(E_dist)
        if self.args.edge_angle:
            h_E.append(E_angles)
        if self.args.edge_direct:
            h_E.append(E_direct)
        # import pdb; pdb.set_trace()
        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        _V = _V.reshape(-1, _V.shape[-1])
        _E = _E.reshape(-1, _E.shape[-1])
        
        mask_pair = (mask[:, :, None] * mask[:,None,:])
        edge_mask = gather_edges(mask_pair[:,:,:,None], E_idx)[:,:,:,0].reshape(-1).bool()

        E = _E[edge_mask]

        d_chains = ((chain_encoding[:, :, None] - chain_encoding[:,None,:])==0).long() #find self vs non-self interaction
        chain_embed = self._idx_embeddings(gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0].reshape(-1))
        chain_embed = chain_embed[edge_mask]

        edge_start_index = torch.arange(N)[None, :, None].repeat(B, 1, E_idx.shape[-1]).to(E_idx.device) + \
            (torch.arange(B, device=E_idx.device) *N).unsqueeze(-1).unsqueeze(-1)
        edge_end_index = E_idx + (torch.arange(B, device=E_idx.device) *N).unsqueeze(-1).unsqueeze(-1)
        edge_index_ = torch.cat((edge_start_index.reshape(-1)[None], edge_end_index.reshape(-1)[None]), dim=0).long()
        edge_index = edge_index_[:, edge_mask] 

        pos_embed = self._positional_embeddings(edge_index, 16) #[B, L, K]
        edge_attr = torch.cat([E, pos_embed, chain_embed], dim=-1)
        batch_id = torch.arange(B, device=edge_index.device)[:, None].repeat(1, N).reshape(-1)
        X_out = X.reshape(B*N, X.shape[-2], X.shape[-1])
        import pdb; pdb.set_trace()
        return X_out, _V, edge_attr, edge_index, batch_id, residue_idx, chain_encoding

        
    def _positional_embeddings(self, E_idx, 
                               num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = E_idx[0]-E_idx[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
    
    def _idx_embeddings(self, d, 
                               num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=d.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
    
    
    