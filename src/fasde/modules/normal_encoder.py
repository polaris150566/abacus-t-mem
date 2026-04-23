import torch
import torch.nn as nn
import numpy as np

class NormalEncoder(nn.Module):
    def __init__(self, y_hidden_dim=128, y_output_dim=128, theta_hidden_dim=128, theta_output_dim=128, num_centers=20):
        super().__init__()
        self.num_centers = num_centers

        self.W_y = nn.Sequential(
            nn.Linear(self.num_centers, y_output_dim),
            nn.LayerNorm(y_output_dim),
        )

        self.W_theta = nn.Sequential(
            nn.Linear(2, theta_output_dim),
            nn.LayerNorm(theta_output_dim),
        )

    def forward(self, X, G, N):
        """
        Args:
            X: 原子坐标 [B, L, M, 3]
            G: 膜仿射变换矩阵 [B, 3, 4]
            N: 膜法向量，模长=halfThickness [B, 3]

        Returns:
            y_emb:    z-depth RBF 嵌入 [B, L, y_output_dim]，膜外残基置零
            theta_emb: 骨架朝向角嵌入 [B, L, theta_output_dim]
            y_norm:   归一化深度 [B, L]，[-1,1] 为膜内
        """
        ################ tensor checks ################
        if not torch.is_tensor(G) or not torch.is_tensor(N):
            raise TypeError(f"G and N must be torch.Tensor, got {type(G)} and {type(N)}")
        if G.dim() != 3 or G.shape[1:] != (3, 4):
            raise ValueError(f"G must have shape [B, 3, 4], got {tuple(G.shape)}")
        if N.dim() != 2 or N.shape[1] != 3:
            raise ValueError(f"N must have shape [B, 3], got {tuple(N.shape)}")
        ################ tensor checks ################
        G = G.to(X.dtype)
        N = N.to(X.dtype)
        gr = G[:, :, :3]  # [B, 3, 3]
        gt = G[:, :, 3:]  # [B, 3, 1]

        CA = X[:, :, 1, :]  # [B, L, 3]
        NM = X[:, :, 0, :]
        C  = X[:, :, 2, :]

        afv = torch.cross(NM - CA, C - CA, dim=2)  # 骨架朝向 [B, L, 3]

        # 仿射变换到膜坐标系
        R_mem = torch.bmm(gr, (CA + gt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)  # [B, L, 3]

        N_exp = N.unsqueeze(1)  # [B, 1, 3]
        half_thick_sq = (N_exp * N_exp).sum(dim=-1)  # [B, 1]

        # 归一化深度 y/halfThickness，[-1,1] 为膜内
        y_norm = (R_mem * N_exp).sum(dim=-1) / half_thick_sq  # [B, L]
        in_mem_mask = ((y_norm >= -1) & (y_norm <= 1)).float()     # [B, L]

        # RBF → Linear → LayerNorm，然后 mask 膜外
        rbf = rbf_encoding(y_norm, num_centers=self.num_centers)  # [B, L, num_centers]
        y_emb = self.W_y(rbf) * in_mem_mask.unsqueeze(-1)              # [B, L, y_output_dim]

        # θ: 骨架朝向与法向量夹角
        cos_t = (afv * N_exp).sum(dim=-1) / (
            N_exp.norm(dim=-1) * afv.norm(dim=-1).clamp(min=1e-6)
        )
        cos_t = cos_t.clamp(-1.0, 1.0)
        sin_t = torch.sqrt(1.0 - cos_t ** 2)
        theta_emb = self.W_theta(torch.stack([sin_t, cos_t], dim=-1))  # [B, L, theta_output_dim]

        return y_emb, theta_emb, y_norm


def rbf_encoding(tensor, num_centers=20):
    """[B, L] -> [B, L, num_centers]"""
    D_min, D_max = -1.0, 1.0
    centers = torch.linspace(D_min, D_max, num_centers, device=tensor.device)
    D_sigma = (D_max - D_min) / num_centers
    diff = tensor.unsqueeze(-1) - centers
    return torch.exp(-diff ** 2 / (2 * D_sigma ** 2))
