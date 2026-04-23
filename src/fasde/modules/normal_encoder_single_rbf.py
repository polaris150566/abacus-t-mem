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
        encoding_mode="rbf",
    ):
        super().__init__()
        self.eps = 1e-6
        self.output_scale = 1.0
        self.abs_depth_noise_std = 0.1
        self.mem_between_mode = mem_between_mode
        self.encoding_mode = encoding_mode

        if encoding_mode == "rbf":
            self.num_centers = int(num_centers)
            self.W_y = nn.Sequential(
                nn.Linear(self.num_centers, y_output_dim),
                nn.LayerNorm(y_output_dim),
            )
        elif encoding_mode == "region_embedding":
            # 3 regions: 0=membrane_inner, 1=transmembrane, 2=membrane_outer
            # null condition (G=0,N=0) -> output zero vector, not through embedding
            self.region_embedding = nn.Embedding(3, y_output_dim)
            self.tm_inner_bound = -8.0
            self.tm_outer_bound = 2.0
        else:
            raise ValueError(f"Unknown encoding_mode: {encoding_mode}, must be 'rbf' or 'region_embedding'")

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

        if self.encoding_mode == "rbf":
            depth_rbf = interface_offset_rbf_encoding(interface_offset, num_centers=self.num_centers)
            y_emb = self.W_y(depth_rbf) * self.output_scale
            is_null = half_thickness < 1e-3  # [B]
            if is_null.any():
                y_emb = y_emb.clone()
                y_emb[is_null] = 0.0
            return y_emb, abs_depth, signed_depth

        else:  # region_embedding
            is_null = half_thickness < 1e-3  # [B]
            if is_null.all():
                y_emb = torch.zeros(X.shape[0], X.shape[1], self.region_embedding.embedding_dim, device=X.device, dtype=X.dtype)
                region_ids = torch.full(interface_offset.shape, -1, dtype=torch.long, device=X.device)
                return y_emb, region_ids, signed_depth

            region_ids = torch.ones_like(interface_offset, dtype=torch.long)  # 1 = transmembrane
            region_ids[interface_offset < self.tm_inner_bound] = 0            # 0 = membrane inner
            region_ids[interface_offset > self.tm_outer_bound] = 2            # 2 = membrane outer
            y_emb = self.region_embedding(region_ids) * self.output_scale
            if is_null.any():
                y_emb[is_null] = 0.0
                region_ids[is_null] = -1
            return y_emb, region_ids, signed_depth


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
