import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# from unicore.utils import one_hot

# from unifold.modules.common import Linear, residual
# from unifold.modules.common import SimpleModuleList
# from unicore.modules import LayerNorm


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class RelativePos(nn.Module):
    def __init__(
        self, 
        emb_dim,
        relpos_k_seq = 32,
        relpos_k_chain = 3,
        relpos_k_segment = 2, 
    ):
        super(RelativePos, self).__init__()
        self.relpos_k_seq = relpos_k_seq
        self.relpos_k_chain = relpos_k_chain
        self.relpos_k_segment = relpos_k_segment

        self.num_seq_bins = 2 * relpos_k_seq + 1
        self.num_chain_bins = 2 * relpos_k_chain + 1
        self.num_segment_bins = 2 * relpos_k_segment + 1

        self.linear_relpos = Linear(
            self.num_seq_bins + self.num_chain_bins + self.num_segment_bins, emb_dim
        )

    @staticmethod
    def rel_pos(residx, max_rel_idx):
        rp = residx[..., None] - residx[..., None, :]
        rp = rp.clip(-max_rel_idx, max_rel_idx) + max_rel_idx
        return rp

    def forward(self, residx, chainidx, segment):
        rel_res_pos = self.rel_pos(residx, self.relpos_k_seq)
        rel_chain_pos = self.rel_pos(chainidx, self.relpos_k_chain)
        rel_segment_pos = self.rel_pos(segment, self.relpos_k_segment)
        relpos = torch.cat([
            F.one_hot(rel_res_pos, num_classes=self.num_seq_bins),
            F.one_hot(rel_chain_pos, num_classes=self.num_chain_bins),
            F.one_hot(rel_segment_pos, num_classes=self.num_segment_bins),
        ], dim=-1)
        pos_embed = self.linear_relpos(relpos.float())

        return pos_embed


class ScoreInputEmbedder(nn.Module):
    def __init__(
        self,
        input_dim_1d: int,
        input_dim_2d: int,
        d_pair: int,
        d_single: int,
        use_prior: bool,
        **kwargs,
    ):
        super(ScoreInputEmbedder, self).__init__()
        self.input_dim_1d = input_dim_1d
        self.input_dim_2d = input_dim_2d
        self.use_prior = use_prior

        self.d_pair = d_pair
        self.d_single = d_single

        self.linear_1d = Linear(input_dim_1d, d_single)
        self.linear_1d_left = Linear(input_dim_1d, d_single)
        self.linear_1d_right = Linear(input_dim_1d, d_single)
        self.linear_2d = Linear(input_dim_2d, d_pair)
        self.linear_cat = Linear(d_single + d_pair + (d_pair if use_prior else 0), d_pair)
        self.pos_embedding = RelativePos(d_pair)

    def get_timestep_embedding(self, t):
        timesteps = t * 999.0
        single_t = get_timestep_embedding(timesteps, self.d_single)
        pair_t = get_timestep_embedding(timesteps, self.d_pair)
        return single_t, pair_t

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        pair_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        single = single.type(self.linear_1d.weight.dtype)
        pair = pair.type(self.linear_1d.weight.dtype)

        single_proj = F.relu(self.linear_1d(single))
        single_left = self.linear_1d_left(single)
        single_right = self.linear_1d_right(single)
        out_add = F.relu(single_left[..., None, :] + single_right[..., None, :, :])
        pair = F.relu(self.linear_2d(pair))

        cat_list = [pair, out_add]
        if self.use_prior:
            pair_cond = pair_cond.type(self.linear_1d.weight.dtype)
            cat_list.append(pair_cond)
        pair = self.linear_cat(
            torch.cat(cat_list, dim=-1)
        )

        return single_proj, pair


class PriorInputEmbedder(nn.Module):
    def __init__(
        self,
        input_dim_1d: int,
        input_dim_2d: int,
        d_pair: int,
        d_single: int,
        **kwargs,
    ):
        super(PriorInputEmbedder, self).__init__()
        self.input_dim_1d = input_dim_1d
        self.input_dim_2d = input_dim_2d

        self.d_pair = d_pair
        self.d_single = d_single

        self.linear_1d = Linear(input_dim_1d, d_single)
        self.linear_1d_left = Linear(input_dim_1d, d_single)
        self.linear_1d_right = Linear(input_dim_1d, d_single)
        self.linear_2d = Linear(input_dim_2d, d_pair)
        self.linear_cat = Linear(d_single + d_pair, d_pair)
        self.pos_embedding = RelativePos(d_pair)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        single = single.type(self.linear_1d.weight.dtype)
        pair = pair.type(self.linear_1d.weight.dtype)

        single_proj = F.relu(self.linear_1d(single))

        single_left = self.linear_1d_left(single)
        single_right = self.linear_1d_right(single)
        out_add = F.relu(single_left[..., None, :] + single_right[..., None, :, :])
        pair = F.relu(self.linear_2d(pair))
        pair = self.linear_cat(
            torch.cat([pair, out_add], dim=-1)
        )

        return single_proj, pair

