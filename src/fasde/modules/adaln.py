"""AdaLN (Adaptive Layer Normalization) for membrane depth conditioning.

Zero-initialized conditioning projection ensures identity behavior at init,
so loading a pretrained checkpoint produces identical outputs before any AdaLN training.
"""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive LayerNorm: predicts per-residue scale and shift from a conditioning signal."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),  # -> (gamma_offset, beta_offset)
        )
        # Zero-init so that gamma_offset=0, beta_offset=0 at start -> identity
        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

    def forward(self, x, cond):
        """
        Args:
            x:    [B, L, hidden_dim]
            cond: [B, L, cond_dim]
        Returns:
            [B, L, hidden_dim]
        """
        gamma_beta = self.cond_proj(cond)  # [B, L, 2*hidden_dim]
        gamma_offset, beta_offset = gamma_beta.chunk(2, dim=-1)
        out = self.norm(x)
        return out * (1 + gamma_offset) + beta_offset


class AdaLNDecLayer(nn.Module):
    """Decoder layer with AdaLN conditioning — a drop-in replacement for DecLayer.

    Replicates DecLayer's architecture exactly but replaces norm1/norm2 with AdaLN.
    Pretrained weights load into the same parameter names (W1, W2, W3, dense, norm1, norm2)
    via the adaln*.norm alias.
    """

    def __init__(self, num_hidden, num_in, cond_dim, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # These are the same as DecLayer — pretrained weights load here
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        # AdaLN wrappers that share the underlying norm with self.norm1/norm2
        self.adaln1 = AdaLN(num_hidden, cond_dim)
        self.adaln2 = AdaLN(num_hidden, cond_dim)
        # Point adaln's internal norm to the same LayerNorm that holds pretrained weights
        self.adaln1.norm = self.norm1
        self.adaln2.norm = self.norm2

    def forward(self, h_V, h_E, cond, mask_V=None, mask_attend=None):
        """Same as DecLayer.forward but with depth conditioning via AdaLN.

        Args:
            h_V:   [B, L, num_hidden]
            h_E:   [B, L, K, num_in]
            cond:  [B, L, cond_dim]  — depth embedding
        """
        # Message aggregation (identical to DecLayer)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        # AdaLN instead of plain LayerNorm
        h_V = self.adaln1(h_V + self.dropout1(dh), cond)

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.adaln2(h_V + self.dropout2(dh), cond)

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    """Duplicated here to avoid circular import from design_utils."""
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


import torch.nn.functional as F


class DepthCrossAttention(nn.Module):
    """Multi-head cross-attention: h_V attends to depth_cond.

    W_O zero-initialized so output=0 at init, preserving pretrained behavior.
    Ref: Flamingo (Alayrac et al., 2022) — gated cross-attention for injecting
    conditioning signal into frozen pretrained models.
    """

    def __init__(self, hidden_dim, cond_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(cond_dim, hidden_dim)
        self.W_V = nn.Linear(cond_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Zero-init W_O -> cross_attn output=0 at init -> identity
        nn.init.zeros_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    def forward(self, h_V, depth_cond, mask=None):
        """
        Args:
            h_V:        [B, L, hidden_dim]
            depth_cond: [B, L, cond_dim]
            mask:       [B, L] node mask (1=valid, 0=pad)
        Returns:
            [B, L, hidden_dim]
        """
        B, L, D = h_V.shape
        H, d = self.num_heads, self.head_dim

        Q = self.W_Q(h_V).view(B, L, H, d).transpose(1, 2)        # [B, H, L, d]
        K = self.W_K(depth_cond).view(B, L, H, d).transpose(1, 2)  # [B, H, L, d]
        V = self.W_V(depth_cond).view(B, L, H, d).transpose(1, 2)  # [B, H, L, d]

        attn = (Q @ K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :].bool(), -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        return self.W_O(out)


class CrossAttnDecLayer(nn.Module):
    """DecLayer + cross-attention to depth between norm1 and FFN.

    Forward signature identical to AdaLNDecLayer: (h_V, h_E, cond, mask_V, mask_attend)
    so the decoder loop in design_utils.py needs no branching.
    """

    def __init__(self, num_hidden, num_in, cond_dim, num_heads=4, dropout=0.1, scale=30):
        super().__init__()
        # === Same as DecLayer — pretrained weights load here ===
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        # === New: cross-attention to depth ===
        self.cross_attn = DepthCrossAttention(num_hidden, cond_dim, num_heads, dropout)
        self.dropout_cross = nn.Dropout(dropout)

    def forward(self, h_V, h_E, cond, mask_V=None, mask_attend=None):
        """
        Args:
            h_V:   [B, L, num_hidden]
            h_E:   [B, L, K, num_in]
            cond:  [B, L, cond_dim] — depth embedding
        """
        # 1. Message aggregation (same as DecLayer)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # 2. Cross-attention to depth (W_O zero-init -> output=0 at start)
        h_V = h_V + self.dropout_cross(self.cross_attn(h_V, cond, mask_V))

        # 3. FFN (same as DecLayer)
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V
