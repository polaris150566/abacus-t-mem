import torch
import torch.nn as nn
import json
from ml_collections import ConfigDict

from .protein_mpnn_utils import (
    ProteinFeatures,
    EncLayer,
    gather_edges,
    gather_nodes,
    cat_neighbors_nodes,
    cat_neighbors_nodes
)

def load_config(path)->ConfigDict:
    return ConfigDict(json.loads(open(path).read()))


class ProteinMPNNEncoder(nn.Module):
    def __init__(self,
        node_features, edge_features,
        hidden_dim, num_encoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super(ProteinMPNNEncoder, self).__init__()

        # Featurization layers
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, prot_X, S, prot_mask, residue_idx, chain_encoding_all):
        pre_E, pre_E_idx = self.features(prot_X, prot_mask, residue_idx, chain_encoding_all, lig_mask=None)
        pre_h_V = torch.zeros((pre_E.shape[0], pre_E.shape[1], pre_E.shape[-1]), device=pre_E.device)
        pre_h_E = self.W_e(pre_E)

        # Encoder is unmasked self-attention
        pre_mask_attend = gather_nodes(prot_mask.unsqueeze(-1),  pre_E_idx).squeeze(-1)
        pre_mask_attend = prot_mask.unsqueeze(-1) * pre_mask_attend
        for layer in self.encoder_layers:
            pre_h_V, pre_h_E = layer(pre_h_V, pre_h_E, pre_E_idx, prot_mask, pre_mask_attend)

        return pre_h_V
