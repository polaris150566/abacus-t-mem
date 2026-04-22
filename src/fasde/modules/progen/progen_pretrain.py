import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from .modeling_progen import ProGenForCausalLM

from tqdm import trange


logger = logging.getLogger(__file__)


class ProgenGPT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        pretrain_checkpiont = self.args.pretrain_progen_checkpoint
        self.model = ProGenForCausalLM.from_pretrained(pretrain_checkpiont)
        if (self.args.modelling_all_codebook):
            if self.args.flatten_all:
                embedder_num = self.args.coarse_code_num
                self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1])
                # if self.args.share_embedder_lmhead:
                #     shared_lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.args.codebook_size + 4)
                #     self.lm_heads = nn.ModuleList()
                #     for _ in range(embedder_num):
                #         self.lm_heads.append(shared_lm_head)
                # else:
                self.lm_heads = nn.ModuleList()
                for _ in range(embedder_num):
                    self.lm_heads.append(
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.args.codebook_size + 4))
            else:
                embedder_num = self.args.codebook_num
                self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.args.depth_n_embd)

            if self.args.share_embedder_lmhead:
                self.codebooks_embedders = nn.ModuleList()
                shared_embedder = nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1])
                for _ in range(embedder_num):
                    self.codebooks_embedders.append(shared_embedder)
            else:
                self.codebooks_embedders = nn.ModuleList()
                for _ in range(embedder_num):
                    self.codebooks_embedders.append(
                        nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1]))
            
        else:
            if self.args.embedding_aatype:
                if self.args.aatype_only:
                    self.aatype_embedder = nn.Sequential(
                        nn.Embedding(20 + 4, self.model.lm_head.weight.shape[-1]),
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1])
                    )
                else:
                    self.codebooks_embedder = nn.Sequential(
                        nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1]//2),
                        nn.Linear(self.model.lm_head.weight.shape[-1]//2, self.model.lm_head.weight.shape[-1]//2),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1]//2, self.model.lm_head.weight.shape[-1]//2)
                    )

                    self.aatype_embedder = nn.Sequential(
                        nn.Embedding(20 + 4, self.model.lm_head.weight.shape[-1]//2),
                        nn.Linear(self.model.lm_head.weight.shape[-1]//2, self.model.lm_head.weight.shape[-1]//2),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1]//2, self.model.lm_head.weight.shape[-1]//2)
                    )

                self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1])

                if self.args.aatype_only:
                    self.aatype_head = nn.Sequential(
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1], 20 + 4)
                    )
                else:
                    self.lm_head = nn.Sequential(
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.args.codebook_size + 4)
                    )

                    self.aatype_head = nn.Sequential(
                        nn.Linear(self.model.lm_head.weight.shape[-1], self.model.lm_head.weight.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(self.model.lm_head.weight.shape[-1], 20 + 4)
                    )

            else:
                self.model.transformer.wte = nn.Embedding(self.args.codebook_size + 4, self.model.lm_head.weight.shape[-1])
                self.model.lm_head = nn.Linear(self.model.lm_head.weight.shape[-1], self.args.codebook_size + 4)

        self._init_weight()        


    def _init_weight(self):
        if self.args.modelling_all_codebook:
            for codebook_embedder in self.codebooks_embedders:
                 codebook_embedder.weight.data.uniform_(-1.0 / (self.args.codebook_size + 4), 1.0 / (self.args.codebook_size + 4))
        else:
            if self.args.embedding_aatype:
                if self.args.aatype_only:
                    self.aatype_embedder[0].weight.data.uniform_(-1.0 / (20 + 4), 1.0 / (20 + 4))
                else:
                    self.codebooks_embedder[0].weight.data.uniform_(-1.0 / (self.args.codebook_size + 4), 1.0 / (self.args.codebook_size + 4))
                    self.aatype_embedder[0].weight.data.uniform_(-1.0 / (20 + 4), 1.0 / (20 + 4))
            else:
                self.model.transformer.wte.weight.data.uniform_(-1.0 / (self.args.codebook_size + 4), 1.0 / (self.args.codebook_size + 4))


    def forward(self, sample):
        if self.args.modelling_all_codebook:
            squeezed_source = sample['net_input']['src_tokens']
            bsz = squeezed_source.shape[0]
            reshaped_source = squeezed_source.reshape(bsz, self.args.codebook_num, -1)
            if self.args.flatten_all:
                embedder_num = self.args.coarse_code_num
                added_source = []
                for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                    residue_codebook_emb = codebook_embedder(reshaped_source[:, codebook_idx])
                    added_source.append(residue_codebook_emb[:, None])
                dimension = added_source[0].shape[-1]
                # import pdb; pdb.set_trace()
                added_source = torch.cat(added_source, 1).permute(0, 2, 1, 3).reshape(bsz, -1, dimension) # BxNCxD
            else:
                added_source = 0
                for codebook_idx, codebook_embedder in enumerate(self.codebooks_embedders):
                    residue_codebook_emb = codebook_embedder(reshaped_source[:, codebook_idx])
                    added_source = added_source + residue_codebook_emb

            dtype = added_source.dtype 
            dec_output = self.model(
                inputs_embeds = added_source, 
                input_bias = 0.0,
                input_bias_layer = -1,
                use_gradient_checkpoint = True,
            ) # BxNCxD

            if self.args.flatten_all:
                reshaped_dec_output = dec_output.logits.reshape(bsz, -1, embedder_num, dimension).to(dtype) # BxNxCxD
                dec_output_logits = []
                for depth_idx, depth_head in enumerate(self.lm_heads): 
                    # import pdb; pdb.set_trace()
                    dec_output_logits.append(depth_head(reshaped_dec_output[:, :, depth_idx])[:, :, None])
                # import pdb; pdb.set_trace()
                dec_output_logits = torch.cat(dec_output_logits, 2) # BxNxCxh
                
                return dec_output_logits.permute(0, 2, 1, 3).float() # BxCxNxh

            else:
                return dec_output.logits
        else:
            source = sample['net_input']['src_tokens']
            if self.args.embedding_aatype:
                bsz = source.shape[0]
                reshaped_source = source.reshape(bsz, 2, -1)

                structure_code = reshaped_source[:, 0]
                aatype_code = reshaped_source[:, 1]
                if self.args.aatype_only:
                    inputs_embeds = self.aatype_embedder(aatype_code)
                else:
                    aatype_rep = self.aatype_embedder(aatype_code)
                    structure_rep = self.codebooks_embedder(structure_code)
                    inputs_embeds = torch.cat([structure_rep, aatype_rep], -1)
                dtype = inputs_embeds.dtype

                dec_output = self.model(
                    inputs_embeds=inputs_embeds, 
                    input_bias = 0.0,
                    input_bias_layer = -1,
                    use_gradient_checkpoint = True,
                ).logits.to(dtype)

                if self.args.aatype_only:
                    aatype_logits = self.aatype_head(dec_output)
                    structure_logits = None
                else:
                    structure_logits = self.lm_head(dec_output)
                    aatype_logits = self.aatype_head(dec_output)

                return {
                    'structure_logits': structure_logits,
                    'aatype_logits': aatype_logits
                }
                
            else:
                dec_output = self.model(
                    source, 
                    input_bias = 0.0,
                    input_bias_layer = -1,
                    use_gradient_checkpoint = True,
                )

                return dec_output.logits


    def sample(self, src_token, max_length, temperature, top_k, ignore_batch=True, aatype_temperature=None):
        if self.args.modelling_all_codebook:
            batch = {'net_input': {'src_tokens': src_token}}
            last_logits = self(batch)[:, -1][:, None]
            return last_logits
        else:
            last_logits = src_token
            bsz = last_logits.shape[0]
            if aatype_temperature is None:
                aatype_temperature = temperature
            for _ in trange(max_length):
                # if the sequence context is growing too long we must crop it at block_size
                batch = {'net_input': {'src_tokens': last_logits}}
                # forward the model to get the logits for the index in the sequence
                if not self.args.embedding_aatype:
                    logits = self(batch)
                    # pluck the logits at the final step and scale by desired temperature
                    logits = logits[:, -1, :] / temperature
                    # optionally crop the logits to only the top k options
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    # apply softmax to convert logits to (normalized) probabilities
                    probs = F.softmax(logits, dim=-1)
                    # sample from the distribution
                    idx_next = torch.multinomial(probs, num_samples=1)
                    # append sampled index to the running sequence and continue
                    last_logits = torch.cat((last_logits, idx_next), dim=1)
                    if ignore_batch:
                        if (idx_next[0].item() == 2):
                            break
                else:
                    last_source = batch['net_input']['src_tokens']
                    reshaped_last_source = last_source.reshape(bsz, 2, -1)

                    str_last_logits = reshaped_last_source[:, 0]
                    seq_last_logits = reshaped_last_source[:, 1]

                    logits_dict = self(batch)
                    structure_logits = logits_dict['structure_logits']
                    aatype_logits = logits_dict['aatype_logits']
                    # pluck the logits at the final step and scale by desired temperature
                    structure_logits = structure_logits[:, -1, :] / temperature
                    aatype_logits = aatype_logits[:, -1, :] / aatype_temperature
                    # optionally crop the logits to only the top k options
                    if top_k is not None:
                        str_v, _ = torch.topk(structure_logits, min(top_k, structure_logits.size(-1)))
                        structure_logits[structure_logits < str_v[:, [-1]]] = -float('Inf')
                        seq_v, _ = torch.topk(aatype_logits, min(top_k, aatype_logits.size(-1)))
                        aatype_logits[aatype_logits < seq_v[:, [-1]]] = -float('Inf')
                    # apply softmax to convert logits to (normalized) probabilities
                    structure_probs = F.softmax(structure_logits, dim=-1)
                    aatype_probs = F.softmax(aatype_logits, dim=-1)
                    # sample from the distribution
                    str_idx_next = torch.multinomial(structure_probs, num_samples=1)
                    seq_idx_next = torch.multinomial(aatype_probs, num_samples=1)
                    # append sampled index to the running sequence and continue
                    str_last_logits = torch.cat((str_last_logits, str_idx_next), dim=1)
                    seq_last_logits = torch.cat((seq_last_logits, seq_idx_next), dim=1)
                    if ignore_batch:
                        if (str_idx_next[0].item() == 2):
                            break

                    last_logits = torch.cat([str_last_logits, seq_last_logits], -1)

            return last_logits[0]



