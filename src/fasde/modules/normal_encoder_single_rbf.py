import torch
import torch.nn as nn


class NormalEncoder(nn.Module):
    def __init__(
        self,
        y_hidden_dim=128,
        y_output_dim=128,
        theta_hidden_dim=128,
        theta_output_dim=128,
        num_centers=20,
        mem_between_mode="add",
        abs_depth_noise_std=0.5,
        theta_noise_std_degrees=10.0,
        embedding_dropout=0.1,
        y_emb_dropout=0.1,
        zero_region_embedding=False,
    ):
        super().__init__()
        self.eps = 1e-6
        self.output_scale = 1.0
        self.abs_depth_noise_std = float(abs_depth_noise_std)
        self.theta_noise_std_degrees = float(theta_noise_std_degrees)
        self.mem_between_mode = mem_between_mode
        self.zero_region_embedding = bool(zero_region_embedding)

        self.num_centers = int(num_centers)
        self.rbf_output_dim = y_output_dim // 4
        self.theta_output_dim = y_output_dim // 4
        self.region_output_dim = y_output_dim - self.rbf_output_dim - self.theta_output_dim
        self.W_y = nn.Sequential(
            nn.Linear(self.num_centers, self.rbf_output_dim),
            nn.LayerNorm(self.rbf_output_dim),
        )
        self.W_theta = nn.Sequential(
            nn.Linear(2, self.theta_output_dim),
            nn.LayerNorm(self.theta_output_dim),
        )
        # 3 regions: 0=membrane_inner, 1=transmembrane, 2=membrane_outer
        self.region_embedding = nn.Embedding(3, self.region_output_dim)
        self.rbf_dropout = nn.Dropout(float(embedding_dropout))
        self.theta_dropout = nn.Dropout(float(embedding_dropout))
        self.region_dropout = nn.Dropout(float(embedding_dropout))
        self.y_emb_dropout = nn.Dropout(float(y_emb_dropout))
        self.tm_inner_bound = -8.0
        self.tm_outer_bound = 2.0

    def forward(self, X, G, N):
        if not torch.is_tensor(X) or not torch.is_tensor(G) or not torch.is_tensor(N):
            raise TypeError(f"X, G and N must be torch.Tensor, got {type(X)}, {type(G)}, {type(N)}")
        if X.dim() != 4 or X.shape[-1] != 3:
            raise ValueError(f"X must have shape [B, L, M, 3], got {tuple(X.shape)}")
        if G.dim() != 3 or G.shape[1:] != (3, 4):
            raise ValueError(f"G must have shape [B, 3, 4], got {tuple(G.shape)}")
        if N.dim() != 2 or N.shape[1] != 3:
            raise ValueError(f"N must have shape [B, 3], got {tuple(N.shape)}")

        G = G.to(device=X.device, dtype=X.dtype)
        N = N.to(device=X.device, dtype=X.dtype)
        gr = G[:, :, :3]
        gt = G[:, :, 3:]

        ca = X[:, :, 1, :]
        r_mem = torch.bmm(gr, (ca + gt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)

        half_thickness = N.norm(dim=-1).clamp(min=self.eps)
        unit_normal = N / half_thickness.unsqueeze(-1)
        signed_depth = (r_mem * unit_normal.unsqueeze(1)).sum(dim=-1)
        abs_depth = signed_depth.abs()
        if self.training and self.abs_depth_noise_std > 0:
            abs_depth = (abs_depth + torch.randn_like(abs_depth) * self.abs_depth_noise_std).clamp_min(0.0)
        interface_offset = abs_depth - half_thickness.unsqueeze(1)

        depth_rbf = interface_offset_rbf_encoding(interface_offset, num_centers=self.num_centers)
        rbf_emb = self.rbf_dropout(self.W_y(depth_rbf))

        n_atom = X[:, :, 0, :]
        c_atom = X[:, :, 2, :]
        virtual_cb = virtual_cb_from_backbone(n_atom, ca, c_atom)
        ca_to_cb = virtual_cb - ca
        ca_to_cb = ca_to_cb / ca_to_cb.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        outward_sign = torch.where(
            signed_depth >= 0,
            torch.ones_like(signed_depth),
            -torch.ones_like(signed_depth),
        )
        outward_normal = unit_normal.unsqueeze(1) * outward_sign.unsqueeze(-1)
        cos_theta = (ca_to_cb * outward_normal).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        if self.training and self.theta_noise_std_degrees > 0:
            theta_noise_std = theta.new_tensor(self.theta_noise_std_degrees * 3.141592653589793 / 180.0)
            theta = (theta + torch.randn_like(theta) * theta_noise_std).clamp(
                min=0.0,
                max=3.141592653589793,
            )
        theta_features = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        theta_emb = self.theta_dropout(self.W_theta(theta_features))

        region_ids = torch.ones_like(interface_offset, dtype=torch.long)  # 1 = transmembrane
        region_ids[interface_offset < self.tm_inner_bound] = 0            # 0 = membrane inner
        region_ids[interface_offset > self.tm_outer_bound] = 2            # 2 = membrane outer
        region_emb = self.region_dropout(self.region_embedding(region_ids))
        if self.zero_region_embedding:
            region_emb = torch.zeros_like(region_emb)
        y_emb = self.y_emb_dropout(torch.cat([rbf_emb, theta_emb, region_emb], dim=-1)) * self.output_scale
        is_null = half_thickness < 1e-3  # [B]
        if is_null.any():
            y_emb = y_emb.clone()
            y_emb[is_null] = 0.0
            region_ids = region_ids.clone()
            region_ids[is_null] = -1
        return {
            "y_emb": y_emb,
            "abs_depth": abs_depth,
            "signed_depth": signed_depth,
            "theta": theta,
            "theta_degrees": theta * theta.new_tensor(180.0 / 3.141592653589793),
            "theta_features": theta_features,
            "rbf_raw": depth_rbf,
            "region_ids": region_ids,
        }


def virtual_cb_from_backbone(n_atom, ca, c_atom):
    b = ca - n_atom
    c = c_atom - ca
    a = torch.cross(b, c, dim=-1)
    return ca + (-0.58273431 * a + 0.56802827 * b - 0.54067466 * c)


def interface_offset_rbf_encoding(distance, num_centers=20):
    centers = torch.linspace(
        -10.0,
        9.0,
        num_centers,
        device=distance.device,
        dtype=distance.dtype,
    )
    sigma = distance.new_tensor(1.0)
    diff = distance.unsqueeze(-1) - centers
    return torch.exp(-(diff ** 2) / (2 * sigma ** 2))
