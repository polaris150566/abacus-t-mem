import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionEncoder(nn.Module):
    def __init__(self, r_emb_dim=128 , output_dim=128, hidden_dim=128, e_hidden_dim = 128,eo_hidden_dim = 64 ):
        super().__init__()

        from ..data.fullatom_dataset import region_dict
        region_dict_len = len(region_dict)

        self.region_embedding = nn.Embedding(region_dict_len, r_emb_dim)

    def forward(self, res_idx, R, chain_masks):
        """对regions进行embedding
        """
        # Embed and process each input
        result_v = self.region_embedding(R)
        pairwise_interactions = None
        return result_v , pairwise_interactions



class RegionDecoder(nn.Module):
    """
    两层前馈网络：Linear → GELU → LayerNorm → Linear
    可选残差连接：x + F(x)
    """

    def __init__(self, d_in = 16, d_hidden = 64, d_out = 64, use_residual = False):
        super().__init__()
        self.use_residual = use_residual

        self.ff = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

        # 若启用残差且维度不一致，则做映射
        if use_residual and d_in != d_out:
            self.residual_proj = nn.Linear(d_in, d_out)
        else:
            self.residual_proj = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_V):
        out = self.ff(h_V)

        if self.use_residual:
            residual = h_V if self.residual_proj is None else self.residual_proj(h_V)
            out = out + residual

        return out