# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os, sys
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import yaml
from .design_utils import ABACUST
from .cmlm_mask import inject_noise, _skeptical_unmasking, _skeptical_unmasking_all

from esm import pretrained
# ESM_PRETRAIN = "/train14/superbrain/lhchen/protein/pretrain/esm2/params/esm2_t33_650M_UR50D.pt"
ESM_PRETRAIN = "/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

DEBUG = False

esm_dict = {
    '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
    'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,
    'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V',
]
restypes = ['<unk>', '<pad>', '<cls>', '<mask>'] + restypes
restype_order = {restype: i for i, restype in enumerate(restypes)}
res_id_to_aatype = {v: k for k, v in restype_order.items()}

af2_to_esm_dict = {
    af2_i: esm_dict[restype]
    for restype, af2_i in restype_order.items()
}
sup_af2_to_esm_dict = {
    i: esm_dict['<mask>']
    for i in np.arange(len(af2_to_esm_dict), 35)
}
af2_to_esm_dict.update(sup_af2_to_esm_dict)

af2_to_esm_convert_indices = torch.zeros(len(af2_to_esm_dict))
for af2_i, esm_i in af2_to_esm_dict.items():
    af2_to_esm_convert_indices[af2_i] = esm_i
af2_to_esm_convert_indices = af2_to_esm_convert_indices.long()
import logging
logger = logging.getLogger(__name__)


def make_prev_token_from_t(time_step, T, B, L, lig_mask, device, output_scores_mask, merged_s):
    """ 在扩散（Diffusion）训练过程中，根据当前时间步 t 生成一个“被部分掩盖的序列”，
        掩盖的比例是time_ste/T
        依据随机
        这个序列将作为下一步的输入（prev_tokens）。
        把 token 变成 [MASK] 或 0,配体保留24
    """
    cur_iter_p = 1 - (time_step / T)# 计算当前保留比例
    # cur_iter_p = torch.ones_like(cur_iter_p)
    output_scores = torch.rand((B, L)).to(device)
    skeptical_mask_prot = _skeptical_unmasking_all(output_scores.masked_fill(~output_scores_mask, 2.0), output_scores_mask, cur_iter_p)
    cur_iter_masked_S = merged_s.masked_fill(skeptical_mask_prot, 0.0)
    cur_iter_masked_S = cur_iter_masked_S.masked_fill(lig_mask.bool(), 24)#把配体调成24

    return cur_iter_masked_S


def converter_from_af2_to_esm(pred_merged_aatype, prot_mask):
    """把 AlphaFold2 的氨基酸编码“翻译”成 ESM 模型能理解的格式，包括：
        Token 映射（AF2 → ESM）。
        添加特殊 token（<cls> 开头，<eos> 结尾）。

    Args:
        pred_merged_aatype (_type_): _description_
        prot_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    device =pred_merged_aatype.device
    esm_no_bos_toks = []
    for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
        device_af2_to_esm_convert_indices = af2_to_esm_convert_indices.to(device)
        b_esm_tokens = device_af2_to_esm_convert_indices[b_af2_tokens]
        esm_no_bos_toks.append(b_esm_tokens)
    esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)

    B, Np = esm_no_bos_toks.shape
    esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0) # 左侧加 <cls>=0
    esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 1) # 右侧加 <pad>=1
    esm_toks[range(B), (prot_mask.sum(1)+1).long()] = 2

    return esm_toks.long(), esm_no_bos_toks.long()
# esm_toks：完整的 ESM 格式序列（含 <cls>, <eos>）。
# esm_no_bos_toks：仅转换后的氨基酸 token（不含特殊 token），可用于后续处理。

def uniform_sampling_t(diffusion_steps, batch_size, device):
    """给定一个整数 T，从 {0, 1, ..., T-1} 里随机挑 batch_size 个数出来。tensor形式返回

    Returns:
        _type_: _description_
    """

    w = np.ones([diffusion_steps])
    p = w / np.sum(w) # p = [1/T, 1/T, ..., 1/T]。
    indices_np = np.random.choice(len(p), size=(batch_size,), p=p)# 按均匀概率从 {0, ..., T-1} 中采样 batch_size 个时间步索引。
    indices = torch.from_numpy(indices_np).long().to(device)# 把 NumPy 数组转成 PyTorch 长整型张量并放到指定设备（CPU/GPU）
    # weights_np = 1 / p[indices_np]
    # weights = torch.from_numpy(weights_np).float().to(device)
    return indices#, weights



class ABACUSTDesigner(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args):
        super().__init__()
        hidden_dim = 128
        # Load unified membrane config
        _mem_config = None
        _mem_config_path = getattr(args, 'mem_config', '')
        if _mem_config_path:
            with open(_mem_config_path, 'r') as f:
                _mem_config = yaml.safe_load(f)
            logger.info(f'Loaded membrane config: {_mem_config_path} -> {_mem_config}')

        # Parse membrane_inject section
        _mi = _mem_config.get('membrane_inject', {}) if _mem_config else {}
        _mem_net_add = _mi.get('mem_net_add', {})
        _mem_net_concat = _mi.get('mem_net_concat', {})
        _adaln = _mi.get('adaln', {})
        _cross_attn = _mi.get('cross_attn', {})

        # Derive old-style params from new config
        if _mem_net_concat.get('enabled', False):
            _derived_mem_between_mode = 'concat'
            _derived_use_depth = True
            _derived_num_centers = _mem_net_concat.get('num_centers', 20)
        elif _mem_net_add.get('enabled', False):
            _derived_mem_between_mode = 'add'
            _derived_use_depth = True
            _derived_num_centers = _mem_net_add.get('num_centers', 20)
        else:
            _derived_mem_between_mode = 'none'
            _derived_use_depth = False
            _derived_num_centers = 20

        if _adaln.get('enabled', False):
            _derived_depth_inject_method = 'adaln'
        elif _cross_attn.get('enabled', False):
            _derived_depth_inject_method = 'cross_attn'
        else:
            _derived_depth_inject_method = 'none'

        # tm_raw: if any membrane inject is enabled, must contain 'depth' for mem_net to be built
        _cli_tm_raw = args.tm_raw
        if _derived_use_depth and 'depth' not in _cli_tm_raw:
            _cli_tm_raw = 'depth_' + _cli_tm_raw
            logger.info(f'Auto-prepended depth to tm_raw: {_cli_tm_raw}')

        _cfg_config = _mem_config.get('cfg', None) if _mem_config else None
        _depth_cfg = None  # no longer used directly

        abacust_config = {
            "num_letters": 35,
            "node_features": hidden_dim,
            "edge_features": hidden_dim,
            "hidden_dim": hidden_dim,
            "num_encoder_layers": args.merge_mpnn_enc_layer_num,
            "num_decoder_layers": args.merge_mpnn_dec_layer_num,
            "augment_eps": args.augment_eps,
            "k_neighbors": 48,
            "vocab": 35,
            "nar": args.nar,
            "lig_neighbor_seq_mask": args.lig_neighbor_seq_mask,
            "ligmpnn_init": args.ligmpnn_init,
            "pre_prot_mode": args.pre_prot_mode,
            "esm_embedder": args.esm_embedder,
            "max_iter_num": args.max_iter_num,
            "embed_unimol_reprs": args.embed_unimol_reprs,
            "encode_mpnn": args.encode_mpnn,
            "tm_raw": _cli_tm_raw,
            "mem_between_mode": _derived_mem_between_mode,
            "num_centers": _derived_num_centers,
            "depth_inject_method": _derived_depth_inject_method,
            "depth_cond_dim": _adaln.get('cond_dim', 128) if _adaln.get('enabled', False) else _cross_attn.get('cond_dim', 128),
            "depth_num_rbf_centers": _adaln.get('num_rbf_centers', 20) if _adaln.get('enabled', False) else _cross_attn.get('num_rbf_centers', 20),
            "cross_attn_num_heads": _cross_attn.get('num_heads', 4),
            "cfg_enabled": _cfg_config.get('enabled', False) if _cfg_config else False,
            "cfg_dropout_prob": _cfg_config.get('dropout_prob', 0.15) if _cfg_config else 0.15,
            "depth_encoding_mode": _mem_config.get('depth_encoding_mode', 'rbf') if _mem_config_path else 'rbf',
            "freeze_modules": _mem_config.get('freeze', None) if _mem_config_path else None,
        }

        # 打印成格式化的 JSON 形式
        import json
        print(json.dumps(abacust_config, indent=4))
        self.abacust = ABACUST(
                            num_letters=35,
                            node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, \
                            num_encoder_layers=args.merge_mpnn_enc_layer_num, num_decoder_layers=args.merge_mpnn_dec_layer_num, augment_eps=args.augment_eps, k_neighbors=48, vocab=35, \
                            nar=args.nar, lig_neighbor_seq_mask=args.lig_neighbor_seq_mask,
                            ligmpnn_init=args.ligmpnn_init, #新的
                            pre_prot_mode=args.pre_prot_mode,
                            esm_embedder=args.esm_embedder, #新的，这个决定了loss怎么计算
                            max_iter_num=args.max_iter_num, #新的
                            embed_unimol_reprs=args.embed_unimol_reprs,
                            encode_mpnn=args.encode_mpnn,
                            tm_raw=_cli_tm_raw,
                            mem_between_mode=_derived_mem_between_mode,
                            num_centers=_derived_num_centers,
                            depth_inject_method=_derived_depth_inject_method,
                            depth_cond_dim=_adaln.get('cond_dim', 128) if _adaln.get('enabled', False) else _cross_attn.get('cond_dim', 128),
                            depth_num_rbf_centers=_adaln.get('num_rbf_centers', 20) if _adaln.get('enabled', False) else _cross_attn.get('num_rbf_centers', 20),
                            cross_attn_num_heads=_cross_attn.get('num_heads', 4),
                            cfg_enabled=_cfg_config.get('enabled', False) if _cfg_config else False,
                            cfg_dropout_prob=_cfg_config.get('dropout_prob', 0.15) if _cfg_config else 0.15,
                            depth_encoding_mode=_mem_config.get('depth_encoding_mode', 'rbf') if _mem_config_path else 'rbf',
                            freeze_modules=_mem_config.get('freeze', None) if _mem_config_path else None,
                            )

        # self.abacust.eval()

        if args.esm_pretrained:
            self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet(args.esm_pretrained)
            logger.info(f'loading ESM model: {args.esm_pretrained}')

        else:
            self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet(ESM_PRETRAIN)
            logger.info(f'loading ESM model: {ESM_PRETRAIN}')

        total_time_step = args.diff_T
        self.T = int(total_time_step)

        if args.pretrained_mpnn_ckpt:
            state = torch.load(args.pretrained_mpnn_ckpt_f, map_location='cpu', weights_only=False)
            model_weights = state["model"]
            model_weights = {k.replace('model.abacust.', ''): v for k, v in model_weights.items() \
                if (k.startswith('model.abacust.') ) }
            self.abacust.load_state_dict(
                model_weights, strict=False
            )

        # Freeze logic is handled inside ABACUST.__init__ via freeze_modules param

        self._freeze_esm_parameter()
        self._log_trainable_parameter_summary()

    def _freeze_esm_parameter(self, ):
        for name, param in self.esm_model.named_parameters():
            param.requires_grad = False
        logger.info(f'freeze parameters of esm model')

    def _log_trainable_parameter_summary(self):
        total_params = 0
        trainable_params = 0
        trainable_names = []

        for name, param in self.named_parameters():
            param_numel = param.numel()
            total_params += param_numel
            if param.requires_grad:
                trainable_params += param_numel
                trainable_names.append(name)

        frozen_params = total_params - trainable_params
        logger.info(
            f'trainable parameters after freeze: {trainable_params:,} / {total_params:,} '
            f'(frozen {frozen_params:,})'
        )
        if trainable_names:
            logger.info('trainable parameter names: ' + ', '.join(trainable_names))


    def forward(
        self,
        batch,
        use_gradient_checkpoint: bool = True,
    ):

        merged_coords = batch['coords']
        merged_s = batch['tokens']
        merged_mask = batch['node_mask']

        B, L = merged_coords.shape[:2]
        device = merged_coords.device
        # merged_chain_mask = torch.ones((B,L)).to(merged_coords.device)
        merged_chain_mask = batch['chainidx'].float().to(merged_coords.device)#原来没有考虑多链的问题，这里考虑了
        merged_residue_idx = batch['residx']

        merged_chain_encoding_all = batch['chainidx']
        randn = torch.randn_like(merged_chain_mask)

        merged_lig_mask = batch['lig_mask']
        lig_node_attr = batch['lig_node_attr']
        lig_edge_index = batch['lig_edge_index']
        lig_edge_attr = batch['lig_edge_attr']
        unimol_reprs = batch['unimol_lig_node_attr']
        torsion_angles=batch['chi_angles']
        alt_chi_angles = batch['alt_chi_angles']
        torsion_angle_mask=batch['chi_mask']
        plip_anno_itype_list = batch['plip_anno_itype_list']
        pdbtm_regions = batch['pdbtm_regions']
        edge_mem_mask = batch['edge_mem_mask']
        G = batch['tmatrix']
        N = batch['normal']
        pdbname = batch["pdbname"]
        try:
            weight = batch['weight']
        except:
            print(batch['weight'])
            print(batch.keys())

        if batch.__contains__('esm_embedding'): #这个其实没用上
            esm_embedding = batch['esm_embedding']
        else:
            esm_embedding = None

        if batch.__contains__('last_iter_aatype'):
            last_iter_aatype = batch['last_iter_aatype']
        else:
            last_iter_aatype = None


        prot_mask = (torch.all(torch.stack([(1-merged_lig_mask), merged_mask], -1), -1)).float()
        output_scores_mask = (merged_s * (1 - prot_mask)) == 0
        cur_timestep = uniform_sampling_t(self.T, B, device).long()# 一列最大时t01的随机变量

        if (np.random.rand() < 0.5):
        # if True:
            last_iter_S_embed = torch.zeros((B, L, 1280)).to(device)
        else:
            with torch.no_grad():
                last_timestep = torch.where(cur_timestep.float() > 0, cur_timestep - 1, 0).long()
                last_iter_null_embed = torch.zeros((B, L, 1280)).to(device)
                last_iter_esm_embed = torch.zeros((B, L, 1280)).to(device)
                last_prev_tokens = make_prev_token_from_t(last_timestep, self.T, B, L, merged_lig_mask, device, output_scores_mask, merged_s)
                # last_prev_tokens = make_prev_token_from_t(self.T, self.T, B, L, merged_lig_mask, device, output_scores_mask, merged_s)




                logits, src_token, log_sc_logits, prev_torsion_mask, pdbtm_regions_mask,edge_mem_mask = self.abacust(
                    merged_coords,
                    merged_s,
                    merged_mask,
                    merged_chain_mask,
                    merged_residue_idx,
                    merged_chain_encoding_all,
                    randn,
                    lig_node_attr, lig_edge_attr, lig_edge_index,
                    pdbtm_regions,
                    G,
                    N,
                    pdbname = pdbname,
                    use_input_decoding_order=False,
                    decoding_order=None,
                    edge_mem_mask = edge_mem_mask,
                    lig_mask=merged_lig_mask,
                    seq_mask=None,
                    esm_embedding=esm_embedding,
                    last_iter_S_embed=last_iter_null_embed,
                    unimol_reprs=unimol_reprs,
                    torsion_angles=torsion_angles,
                    alt_chi_angles=alt_chi_angles,
                    torsion_angle_mask=torsion_angle_mask,
                    prev_tokens=last_prev_tokens, #新的
                    plip_anno_itype_list=plip_anno_itype_list #新的
                    )

                tokens_mask = (last_prev_tokens == 0)
                tokens_mask = tokens_mask.int()

                #################################################################################
                # token_mask = (merged_mask * (1-merged_lig_mask)).int()
                # tgt_token = merged_s * token_mask
                # logits = logits * token_mask[..., None]

                # logits_f = logits.float()
                # aatype_pred = torch.argmax(logits_f, dim=-1)
                # aatype_acc = torch.sum((aatype_pred == tgt_token).float() * token_mask)/(token_mask.sum() + 1e-6)
                # control_acc = torch.sum((last_prev_tokens == tgt_token).float() * token_mask)/(token_mask.sum() + 1e-6)
                # # import pdb;pdb.set_trace()
                # print(f"aatype_acc:{aatype_acc}")
                # print(f"control_acc:{control_acc}")
                ##################################################################################



                # 贪心采样算法将之前一轮生成的logit转化成氨基酸，之后再将原来就确定的部分替换成merged_s
                #之后将氨基酸的序列转换成esm格式之后输入esm模型中
                pred_aatype = torch.argmax(logits[..., 4: 24], -1) + 4
                pred_merged_aatype = tokens_mask * pred_aatype + (1 - tokens_mask) * merged_s
                input_esm_tokens, raw_esm_tokens = converter_from_af2_to_esm(pred_merged_aatype, prot_mask)
                # 用 ESM 预训练模型对输入的蛋白序列做前向推理，提取第 33 层的隐藏表征，并去掉首尾的 <cls> 和 <eos> token，得到每个氨基酸的嵌入向量。
                # output：是一个字典，包含：
                # {
                #     "representations": {33: tensor of shape (B, L+2, d)}  # d=1280 for ESM-2-650M
                # }
                output = self.esm_model(input_esm_tokens, repr_layers=[33], return_contacts=False)
                esm_rep_ = output['representations'][33][:, 1:-1].detach()

            last_iter_esm_embed = torch.where((prot_mask == 1)[..., None], esm_rep_, last_iter_esm_embed)
            last_iter_S_embed = torch.where((cur_timestep == 0)[:, None, None], last_iter_null_embed, last_iter_esm_embed)#生成的序列进行esm向前传播之后加上掩码

        cur_prev_tokens = make_prev_token_from_t(cur_timestep, self.T, B, L, merged_lig_mask, device, output_scores_mask, merged_s)#去掉不准的氨基酸

        logits, src_token, log_sc_logits, prev_torsion_mask, pdbtm_regions_mask,edge_mem_mask = self.abacust(
            merged_coords, merged_s, merged_mask, merged_chain_mask, merged_residue_idx, merged_chain_encoding_all, randn,
            lig_node_attr, lig_edge_attr, lig_edge_index,
            pdbtm_regions,
            G,
            N,
            pdbname = pdbname,
            use_input_decoding_order=False,
            decoding_order=None,
            edge_mem_mask = edge_mem_mask,
            lig_mask=merged_lig_mask, seq_mask=None,
            esm_embedding=esm_embedding,
            last_iter_S_embed=last_iter_S_embed,
            unimol_reprs=unimol_reprs,
            torsion_angles=torsion_angles, alt_chi_angles=alt_chi_angles, torsion_angle_mask=torsion_angle_mask,
            prev_tokens=cur_prev_tokens,
            plip_anno_itype_list=plip_anno_itype_list
            )

        sample_cfgs = {
            "weight": weight,
            "edge_mem_mask":edge_mem_mask
            }

        return logits, src_token, log_sc_logits, prev_torsion_mask, pdbtm_regions_mask,sample_cfgs


    def sample(
        self,
        batch,
        prev_output_tokens = None,
        target_shuffle_index: Tensor = None,
        target_inv_shuffle_index: Tensor = None,
        return_all_hiddens: bool = False,
        features_only: bool = False,
        use_gradient_checkpoint: bool = True,
        tmp = 0.1
    ):
        coords = batch['coords']
        confidence = batch['confidence']
        positions = batch['residx']
        padding_mask = batch['padding_mask'] = (batch['node_mask'] == 0.0)
        encoder_out = self.encoder(batch, coords, padding_mask, confidence, positions,
            return_all_hiddens=return_all_hiddens,use_gradient_checkpoint=use_gradient_checkpoint)
        max_len = batch['seg_len'][:,0].max()
        lig_mask = batch['lig_mask'].bool()
        rec_mask = batch['node_mask'] * (1 - lig_mask.float())
        # prev_output_tokens = (prev_output_tokens*rec_mask) [:,:max_len-1].long()
        gg = prev_output_tokens*rec_mask + (1-rec_mask)*torch.ones_like(prev_output_tokens)
        prev_output_tokens = gg[:,:max_len-1].long()
        positions = (positions*rec_mask)[:,:max_len].long()
        target_shuffle_index = (target_shuffle_index*rec_mask)[:,:max_len].long()
        extra = None
        x = encoder_out['encoder_out'][0]
        edge_index = encoder_out['edge_index']
        edge_embeddings = encoder_out['edge_embeddings']
        L = batch['seg_len'][0,0]
        tokens = torch.tensor([[1,33]]).long().to(x.device)

        for i in range(2,L):
            x_ = x[:,:i]
            emb_s = self.embed_s(tokens)
            emb_s = torch.concat([emb_s[:,-1:], emb_s[:,:-1]],dim=1)
            x_ = torch.concat([emb_s, x_], dim=-1)
            mask = (edge_index[0]<i)&(edge_index[1]<i)
            edge_index_ = edge_index[:,mask]
            edge_embeddings_ = edge_embeddings[mask, :]
            x_ = self.decoder(x_,  edge_index_, edge_embeddings_)
            logits = self.cls(x_)
            if tmp<=0:
                token = logits[:,-1].argmax()
            else:
                probs = F.softmax(logits[:,-1]/tmp, dim=-1)
                token = torch.multinomial(probs, 1)[0,0]
            tokens = torch.concat([tokens, token[None,None]],dim=-1)
        xx =  batch['tokens'][0,1:L-1]==tokens[0,2:]
        # print(xx.float().mean())
        return tokens[:,2:]
