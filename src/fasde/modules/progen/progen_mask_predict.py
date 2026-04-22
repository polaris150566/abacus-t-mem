import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import ml_collections
import json

from .modeling_progen import ProGenForCausalLM
from .configuration_progen import ProGenConfig

_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")

def load_config(path)->ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads(open(path).read()))


logger = logging.getLogger(__file__)


class ProgenMask(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.gamma = self.gamma_func(self.args.gamma)

        pretrain_checkpiont = self.args.pretrain_progen_checkpoint
        progen_config_f = self.args.pretrain_progen_config
        if os.path.isdir(pretrain_checkpiont):
            self.model = ProGenForCausalLM.from_pretrained(pretrain_checkpiont)
        else:
            # import pdb; pdb.set_trace()
            config_f = f'{progen_config_f}'
            model_arg= load_config(config_f)
            depth_model_args = dict(
                vocab_size=self.args.codebook_size + 4,
                n_positions = model_arg.n_positions,
                n_ctx=model_arg.n_embd, n_embd=model_arg.n_embd,
                n_layer=model_arg.n_layer, n_head=model_arg.n_head, 
                rotary_dim=model_arg.rotary_dim, 
                gradient_checkpointing=True,
                embd_pdrop=model_arg.embd_pdrop)
            spatial_config = ProGenConfig(**depth_model_args)
            self.model = ProGenForCausalLM(spatial_config)

        if self.args.modelling_all_codebook:
            self.codebooks_embedders = nn.ModuleList()
            for _ in range(self.args.codebook_num):
                self.codebooks_embedders.append(
                    nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1]))
            self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.args.depth_n_embd)
        else:
            self.codebooks_embedders = nn.ModuleList()
            for _ in range(self.args.target_code_idx+1):
                self.codebooks_embedders.append(
                    nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1]))
            self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.args.codebook_size + 4)

        self.mask_label_embed = nn.Embedding(2, self.model.lm_head.weight.shape[-1])

        self._init_weight()        


    def _init_weight(self):
        for codebook_embedder in self.codebooks_embedders:
                codebook_embedder.weight.data.uniform_(-1.0 / (self.args.codebook_size + 4), 1.0 / (self.args.codebook_size + 4))


    def forward(self, sample):
        squeezed_source = sample['net_input']['src_tokens']
        bsz = squeezed_source.shape[0]
        reshaped_source = squeezed_source.reshape(bsz, self.args.codebook_num, -1)
        res_num = reshaped_source.shape[2]
        device = reshaped_source.device
        if self.args.modelling_all_codebook:
            unmasked_source = 0
            for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                residue_codebook_emb = codebook_embedder(reshaped_source[:, codebook_idx])
                unmasked_source = unmasked_source + residue_codebook_emb
            masked_source = self.codebooks_embedders[0](reshaped_source[:, 0]) ######
        else:
            unmasked_source = 0
            masked_source = 0
            for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                residue_codebook_emb = codebook_embedder(reshaped_source[:, codebook_idx])
                unmasked_source = unmasked_source + residue_codebook_emb
                if (codebook_idx < self.args.target_code_idx):
                    masked_source = masked_source + residue_codebook_emb

        r = math.floor(self.gamma(np.random.uniform()) * res_num)
        samp = torch.rand((bsz, res_num), device=device).topk(r, dim=1).indices
        mask = torch.zeros((bsz, res_num), dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=samp, value=True)

        added_source = mask * unmasked_source + (~mask) * masked_source

        added_source = added_source + self.mask_label_embed(mask[..., 0].long())

        dec_output = self.model(
            inputs_embeds = added_source, 
            input_bias = 0.0,
            input_bias_layer = -1,
            use_gradient_checkpoint = True,
            mask_bias=False
        )

        return dec_output.logits


    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError


    @torch.no_grad()
    def sample(self, src_tokens, T=11, choice_temperature=1.0, depth_model=None, au_templerature=1.0, au_topk=None):
        if self.args.modelling_all_codebook:
            assert len(src_tokens.shape) == 2
            masked_source = self.codebooks_embedders[0](src_tokens)
            assert depth_model is not None
        else:
            assert len(src_tokens.shape) == 3
            masked_source = 0
            for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                if (codebook_idx < self.args.target_code_idx):
                    residue_codebook_emb = codebook_embedder(src_tokens[:, codebook_idx])
                    masked_source = masked_source + residue_codebook_emb

        bsz, res_num = masked_source.shape[:2]
        device = masked_source.device
        dtype = masked_source.dtype

        mask = torch.zeros(bsz, res_num).long().to(device)
        # mask[:, 0] = torch.ones(bsz, 1).long().to(device)
        unknown_number_in_the_beginning = torch.sum(mask == 0, dim=-1) # 0 mask, 1 unmask
        last_ids = None  # [8, 257]
        # last_ids = torch.zeros(bsz, res_num).long().to(device)

        for t in range(T):
            if last_ids is not None:
                if self.args.modelling_all_codebook:
                    unmasked_source = 0
                    for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                        residue_codebook_emb = codebook_embedder(last_ids[:, codebook_idx])
                        unmasked_source = unmasked_source + residue_codebook_emb
                    # import pdb; pdb.set_trace()
                    added_source = mask[..., None] * unmasked_source + (~mask)[..., None] * masked_source
                else:
                    added_source = torch.where(mask[..., None] == 1, 
                        masked_source + self.codebooks_embedders[-1](last_ids), masked_source)
            else:
                added_source = masked_source

            added_source = added_source + self.mask_label_embed(mask)
            dec_output = self.model(
                inputs_embeds = added_source, 
                input_bias = 0.0,
                input_bias_layer = -1,
                use_gradient_checkpoint = True,
                mask_bias=False
            )

            unknown_map = (1 - mask).bool() # 1 mask, 0 unmask
            
            if not self.args.modelling_all_codebook:
                
                logits = dec_output.logits
                sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
                if last_ids is not None:
                    sampled_ids = torch.where(unknown_map, sampled_ids, last_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]
                probs = F.softmax(logits, dim=-1)
                selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
                selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)

            else:
                sampled_ids, probs = depth_model.generate_au(dec_output.logits.to(dtype), au_templerature, au_topk) # BxCxL, BxCxLxD
                
                if last_ids is not None:
                    # import pdb; pdb.set_trace()
                    sampled_ids = torch.where(unknown_map[:, None], sampled_ids, last_ids[:, 1:]) # BxCxL
                batchsize, codbk_num = sampled_ids.shape[:2]
                selected_probs = torch.stack([ 
                        torch.squeeze(torch.take_along_dim(probs[:, cod_idx], 
                        torch.unsqueeze(sampled_ids[:, cod_idx], -1), -1), -1) 
                        for cod_idx in range(codbk_num) 
                    ]) # CxBxL
                # import pdb; pdb.set_trace()
                selected_probs = torch.mean(selected_probs, 0) # CxBxL -> BxL
                selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)

            
            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio) # 1., 0.99, 0.96, 0.92 ..., 0.13
            # import pdb; pdb.set_trace()
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            mask = self.mask_by_random_topk(mask_len, selected_probs, temperature=choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            # import pdb; pdb.set_trace()
            last_ids = torch.cat([src_tokens[:, None], sampled_ids], 1)

        return last_ids


    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1) # from low to high
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1) # mask_len from high to low
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        mask = (1 - masking.long())
        return mask

