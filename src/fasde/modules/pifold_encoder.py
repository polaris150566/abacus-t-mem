from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from .pifold.model import PiFoldModel


@dataclass
class PiFoldConfig:
    display_step: int = 10
    d_model: int = 128
    hidden_dim: int = 128
    node_features: int = 128
    edge_features: int = 128
    k_neighbors: int = 30
    dropout: float = 0.1
    num_encoder_layers: int = 10
    updating_edges: int = 4
    node_dist: int = 1
    node_angle: int = 1
    node_direct: int = 1
    edge_dist: int = 1
    edge_angle: int = 1
    edge_direct: int = 1
    virtual_num: int = 3

    n_vocab: int = 32
    use_esm_alphabet: bool = False

_default_cfg = PiFoldConfig()

class PiFoldEncoder(nn.Module):
    def __init__(self,) :
        super().__init__()

        self.model = PiFoldModel(args=_default_cfg)


    def forward(self, X, mask, prev_tokens):
        feats = self.model(
            X=X,
            mask=mask.float(),
            S=prev_tokens,
            lengths=None)

        return feats
