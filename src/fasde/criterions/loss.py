import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import lengths_to_mask
from fairseq.dataclass import FairseqDataclass
# from ..data.fullatom_dataset import region_dict, tmtype_dict, reverse_region_dict


# region_dict = {
#     '1': 1,
#     '2': 2,
#     'H': 3,
#     'B': 4,
#     'C': 5,
#     'L': 6,
#     'I': 7,
#     'U': 8,
#     '0': 0,#这个是录入数据的时候填充的，在将xml转换成json的时候，region中没有但是序列里有的aa被标记为0
#     'F': 9
#     #先这么记着，pdbtm里没给f啥意思，但是xml里面有（1ar1)之后抽空查一下怎么回事。。。
# }

# from protein_utils.pdbtm_data_parser import Pdbtm_parser
# region_dict = Pdbtm_parser._REGION_CODE_MAP

region_dict = {
    "1": 1,  # Side-one (extracellular/cytosolic)
    "2": 2,  # Side-two (opposite side)
    "H": 3,  # Transmembrane helix
    "B": 4,  # Transmembrane beta barrel
    "F": 5,  # Interfacial helix
    "L": 6,  # Re-entrant loop
    "N": 7,  # Beta-barrel inside
    "3": 8,  # Periplasm / inter-membrane space (double-membrane)
    "P": 9,  # False-positive membrane (fragment analysis)
    "R": 10, # False-negative membrane (fragment analysis)
    "U": 0  #unknown/other
}
reverse_region_dict = {v: k for k, v in region_dict.items()}
# 1, for side one
# H|B|C|I|L, for membrane embedded
# H for alpha helix,
# B for beta strand,
# C for coiled structure
# L for membrane embedded region not crossing the membrane (loop)
# I for membrane embedded region not interacting with lipids, polypeptid segment inside a beta barrel) :
# 2 for side two :
# U for unknown / missing residue


DEBUG = False

logger = logging.getLogger(__name__)

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


def discrete_torsion(dist, edges=None):
    """对输入的任意形状的张量（通常为(B,L,N)形状）之后根据输入的edge（一个数组或者元组）确定的最大最小值一个要分成X区间，之后将扭转角缩放到（0-X）范围内返回。
    超过范围的会进行裁剪

    Returns:
        _type_: 同样类型的张量
    """
    min_, max_, nbin_ = edges
    dist = (dist - min_) * nbin_ / (max_ - min_)
    dist = dist.int()
    dist = torch.clip(dist, 0, nbin_-1)
    return dist


@dataclass
class CriterionConfig(FairseqDataclass):
    loss_reduction: str = field(default="sum", metadata={"help": "loss reduction"})


@register_criterion("diff_full_atom_criterion", dataclass=CriterionConfig)
class DiffFullAtomCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.loss_reduction = 'sum'

        # 从 mem_config yaml 读取 aa_loss_weights（与 mem_config 机制复用）
        # 格式示例：
        #   aa_loss_weights:
        #     W: 2.0
        #     Y: 2.0
        aa_w = {}
        mem_config_path = getattr(task.args, 'mem_config', '')
        if mem_config_path:
            import yaml
            with open(mem_config_path, 'r') as f:
                mem_cfg = yaml.safe_load(f)
            aa_w = mem_cfg.get('aa_loss_weights', {}) or {}

        # 构建 per-AA loss 权重 tensor，按 restypes 索引（共24个token）
        # restypes = ['<unk>','<pad>','<cls>','<mask>','A','R','N','D','C','Q','E','G',
        #              'H','I','L','K','M','F','P','S','T','W','Y','V']
        weight_vec = [aa_w.get(restypes[i], 1.0) for i in range(len(restypes))]
        self.register_buffer(
            "aa_loss_weight_table",
            torch.tensor(weight_vec, dtype=torch.float32)
        )


    def converter_from_af2_to_esm(self, pred_merged_aatype, rec_len, prot_mask):
        """将预测的氨基酸类型（pred_merged_aatype）从AlphaFold2（AF2）格式转换为ESM模型所需的格式。
        prot_mask用于标识哪些位置是蛋白质

        Args:
            pred_merged_aatype (_type_): 数组，输入的氨基酸
            rec_len (_type_): 每个蛋白质多长
            prot_mask (_type_): 标记哪里是蛋白质

        Returns:
            一个张量，形状比输入大: 会在起始和终止的地点添加esm标记（0，1，2）表示序列开始和结束
        """
        device =pred_merged_aatype.device
        esm_no_bos_toks = []
        for b_idx, b_af2_tokens in enumerate(pred_merged_aatype):
            b_esm_tokens = af2_to_esm_convert_indices[b_af2_tokens]
            esm_no_bos_toks.append(b_esm_tokens)

        esm_no_bos_toks = torch.stack(esm_no_bos_toks, 0).to(device)
        esm_no_bos_toks = (1 - prot_mask) + prot_mask * esm_no_bos_toks

        esm_toks = F.pad(esm_no_bos_toks, (1, 0, 0, 0), 'constant', 0)
        esm_toks = F.pad(esm_toks, (0, 1, 0, 0), 'constant', 2)
        # for b_idx, b_prot_len in enumerate(rec_len):
        #     esm_toks[b_idx][b_prot_len.item()+1] = 2

        return esm_toks.long()


    def forward(self, model, sample):
        """输入模型，从这个模型中提取相关的架构参数，如是否使用非自回归方式，lig_neighbor相关掩码，以及是否使用esm嵌入，tensorbin取多少等
        之后根据输入的sample，以及预测的相关数据计算氨基酸预测的交叉熵损失，扭转角以及备用扭转角离散化之后的交叉熵损失，并将其进行加和

        Args:
            model (_type_): _description_
            sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        nar = model.model.abacust.nar
        # nar = model.model.proteinmpnn.nar
        pred_lig_neighbor_seq_mask = model.model.abacust.lig_neighbor_seq_mask
        # esm_embedder = None
        esm_embedder = model.model.abacust.esm_embedder   #这个过两天确认一下，一个是abacust/mpnn的事情，另一个是esmembedder在哪里的事情
        torsion_bins = model.model.abacust.torsion_bins
        unavail_torsion_loss_weight = model.model.abacust.torsion_loss_weight
        avail_torsion_loss_weight = model.model.abacust.avail_torsion_loss_weight

        if DEBUG:
            logger.info(f'esm_embedder:{esm_embedder}')


        #如果启用了ESM嵌入器，则调用forward_iter方法进行迭代训练，并直接返回其结果
        if esm_embedder:#false
            loss, sample_size, logging_output = self.forward_iter(model, sample)
            return loss, sample_size, logging_output

        else:
            logits, src_token, log_sc_logits, avail_torsion_mask, pdbtm_regions_mask, sample_cfgs = model(sample)
            edge_mem_mask = sample_cfgs["edge_mem_mask"]
            weight = sample_cfgs["weight"]

            logits_f = logits.float()
            log_sc_logits_f = log_sc_logits.float()

            tgt_token = sample['tokens']
            chi_angles = sample['chi_angles']
            disc_chi_angles = discrete_torsion(chi_angles, (-np.pi, np.pi, torsion_bins))
            alt_chi_angles = sample['alt_chi_angles']
            disc_alt_chi_angles = discrete_torsion(alt_chi_angles, (-np.pi, np.pi, torsion_bins))
            torsion_angle_mask = sample['chi_mask']
            unavail_torsion_mask = torsion_angle_mask - avail_torsion_mask
            #这里将标记可用的扭转角和实际可用的扭转角进行相减，仅保留可以使用的扭转角，作为掩码返回
            # 输入数据sample
            # 'tokens'：形状为 [B, L]，表示目标氨基酸类型序列
            # 'chi_angles'：形状为 [B, L, N]，表示扭转角的连续值
            # 'alt_chi_angles'：形状为 [B, L, N]，表示备用扭转角的连续值
            # 'chi_mask'：形状为 [B, L]，表示扭转角掩码
            # 'avail_torsion_mask'：形状为 [B, L]，表示可用扭转角掩码
            # 'lig_mask'：形状为 [B, L]，表示配体掩码
            # 'node_mask'：形状为 [B, L]，表示节点掩码


            lig_mask = sample['lig_mask']
            node_mask = sample['node_mask']

            prot_mask = (torch.all(torch.stack([(1-lig_mask), node_mask], -1), -1)).float()
            torsion_angle_mask = torsion_angle_mask * prot_mask[..., None]


            if not nar:
                if pred_lig_neighbor_seq_mask:
                    tokens_mask = torch.all(torch.stack([prot_mask], -1), -1)
                else:
                    tokens_mask = prot_mask
                tokens_mask = tokens_mask.int()
            else:
                #如果使用了自回归编码的手段，直接检验src——token是不是0，如果是的直接加上掩码之后转化为整形格式
                #src_token = 0意味着这个位置被mask掉了，是新设计的，需要参与loss运算
                tokens_mask = (src_token == 0)
                tokens_mask = tokens_mask.int()

            ########################################################
            tokens_mask = (node_mask * (1-lig_mask)).int()
            ########################################################

            tgt_token = tgt_token * tokens_mask
            logits = logits * tokens_mask[..., None]

            logits_f = logits.float()
            sample_size = tokens_mask.sum()
            batch_size = logits.shape[0]

            ca_coords = sample['coords'][:, :, 1]
            lig_ca_coords = sample['lig_mask'][..., None] * ca_coords
            prot_ca_coords = ((1 - sample['lig_mask']) * sample['node_mask'])[..., None] * ca_coords
            all_atom_dist = torch.sqrt(torch.sum((prot_ca_coords[:, :, None] - lig_ca_coords[:, None])**2, -1)) # B x Np+Nl x Np+Nl
            all_atom_dist_mask = ((1 - sample['lig_mask']) * sample['node_mask'])[:, :, None] * sample['lig_mask'][:, None]
            all_atom_dist = all_atom_dist + (1 - all_atom_dist_mask)* 1e6
            lig_neighbor_mask = torch.any(all_atom_dist < 10.0, dim=-1).long()
            pocket_weight = 0.5
            lig_neighbor_weight = (lig_neighbor_mask * pocket_weight + ((1 - sample['lig_mask']) * sample['node_mask']))


            #######################################################################
            #计算跨膜区的掩码
            tm_region_mask_template = {k: 1 if k in {'H','B',"F","R"} else 0 for k in region_dict.keys()}
            tm_region_table = torch.tensor(
                    [ tm_region_mask_template[reverse_region_dict[i]] for i in range(len(region_dict))],
                    dtype=torch.float32,
                    device=tokens_mask.device
                )
            tm_region_mask = tm_region_table[pdbtm_regions_mask]#标记每一个未知是不是在跨膜区
            nontm_region_mask = 1 - tm_region_mask
            #######################################################################
            # regions的损失加权
            char2weight = {k: 1 if k in {'H','B',"F",'R'} else 1.0 for k in region_dict.keys()}
            # 2. 把映射表转成 tensor，方便一次性索引，假设字符的整数编码就是 region_dict 的值（0~9）
            weight_table = torch.tensor(
                    [char2weight[reverse_region_dict[i]] for i in range(len(region_dict))],
                    dtype=torch.float32,
                    device=tokens_mask.device
                )
            region_weight = weight_table[pdbtm_regions_mask]  # (B, L) #基于regions的权重，可以直接拿来用
            ############################################################################
            #在膜水交界的地方设立权重
            # import pdb;pdb.set_trace()
            edge_mem_weight = torch.where(edge_mem_mask == 1, torch.tensor(1).cuda(), torch.tensor(1.0).cuda()).float()

            # print(edge_mem_weight)
            # print(final_weight)
            #############################################################################

            #计算氨基酸类型的交叉熵损失
            raw_loss = F.cross_entropy(
                torch.reshape(logits_f, (-1, logits_f.size(-1))),
                torch.reshape(tgt_token, (-1,)),
                reduction = 'none'
                ).reshape(tgt_token.shape)

            # per native-AA 权重：根据原始氨基酸类别对 loss 加权
            aa_weight = self.aa_loss_weight_table.to(tgt_token.device)[
                tgt_token.clamp(0, len(restypes) - 1)
            ]  # (B, L)

            loss = raw_loss * aa_weight * lig_neighbor_weight * region_weight * edge_mem_weight  #加权重
            aatype_reduced_loss = (loss * tokens_mask).sum()/(tokens_mask.sum() + 1e-6)


            #氨基酸类型的预测准确率（先通过求最大值之后所得的预测与实际的氨基酸做差，tgttoken是实际的氨基酸
            aatype_pred = torch.argmax(logits_f, dim=-1)
            aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask)/(tokens_mask.sum() + 1e-6)
            # print(f"aatype_acc:{aatype_acc}")

            #############################################################################################
            tm_aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask * tm_region_mask)/(tm_region_mask.sum() + 1e-6)
            tm_aatype_loss = (loss * tokens_mask *  tm_region_mask).sum()/(tm_region_mask.sum() + 1e-6)
            #跨膜区氨基酸准确度和损失

            nontm_aatype_acc = torch.sum((aatype_pred == tgt_token).float() * tokens_mask * nontm_region_mask)/(nontm_region_mask.sum() + 1e-6)
            nontm_aatype_loss = (loss * tokens_mask * nontm_region_mask).sum()/(nontm_region_mask.sum() + 1e-6)
            #跨膜区氨基酸准确度

            # W/Y 召回率：在 tgt 为 W(21) 或 Y(22) 的位置上，预测正确的比例
            wy_mask = ((tgt_token == restype_order['W']) | (tgt_token == restype_order['Y'])).float() * tokens_mask
            wy_recall = (aatype_pred == tgt_token).float() * wy_mask
            wy_recall = wy_recall.sum() / (wy_mask.sum() + 1e-6)
            ############################################################################################
            #计算regions序列标注的交叉熵损失
            # region_logits_f = torch.zeros()
            # regions_loss = F.cross_entropy(
            #     torch.reshape(region_logits_f,(-1,region_logits_f.size(-1))),
            #     torch.
            #     )



            ############################################################################################3


            #扭转角的交叉熵损失：使用交叉熵损失对离散化的扭转角进行交叉熵损失
            torsion_loss = F.cross_entropy(
                torch.reshape(log_sc_logits_f, (-1, log_sc_logits_f.size(-1))),
                torch.reshape(disc_chi_angles.long(), (-1,)), reduction = 'none').reshape(disc_chi_angles.shape)

            #对扭转角进行交叉熵损失计算
            alt_torsion_loss = F.cross_entropy(
                torch.reshape(log_sc_logits_f, (-1, log_sc_logits_f.size(-1))),
                torch.reshape(disc_alt_chi_angles.long(), (-1,)), reduction = 'none').reshape(disc_alt_chi_angles.shape)

            torsion_loss = torch.min(torch.stack([torsion_loss, alt_torsion_loss]), dim=0)[0]
            #扭转角和备用扭转角分别计算交叉熵损失，然后取两者中的较小值作为最终的扭转角损失。
            torsion_loss = torsion_loss * lig_neighbor_weight[..., None] * region_weight[..., None] * edge_mem_weight[...,None]  #加权重
            avail_reduced_torsion_loss = (torsion_loss * avail_torsion_mask).sum()/(avail_torsion_mask.sum() + 1e-6)
            #将 torsion_loss 与 avail_torsion_mask 相乘，过滤掉不可用扭转角的损失，然后求和并归一化（除以掩码的总和加上一个小的常数以避免除零）
            unavail_reduced_torsion_loss = (torsion_loss * unavail_torsion_mask).sum()/(unavail_torsion_mask.sum() + 1e-6)

            #也是一样，将概率最大的作为预测的扭转角之后跟sample中的那啥进行对比，之后得出准确率，之后使用备用的扭转角将该过程重复一遍
            torsion_pred = torch.argmax(log_sc_logits_f, dim=-1)
            torsion_acc = (torsion_pred == disc_chi_angles).float() * torsion_angle_mask
            alt_torsion_acc = (torsion_pred == disc_alt_chi_angles).float() * torsion_angle_mask
            torsion_acc = torch.max(torch.stack([torsion_acc, alt_torsion_acc]), dim=0)[0]

            reduced_torsion_acc = (torsion_acc * torsion_angle_mask).sum()/(torsion_angle_mask.sum() + 1e-6)
            #同样，去除掩码位置的loss，之后归一化

            reduced_torsion_loss = avail_reduced_torsion_loss * avail_torsion_loss_weight + unavail_reduced_torsion_loss * unavail_torsion_loss_weight
            #将可用的与不可用的位置的扭转角做和

            reduced_loss = aatype_reduced_loss + reduced_torsion_loss #+ reduced_hydro_loss*0.4
            #将氨基酸预测的loss和扭转角的loss加起来

            # import pdb;pdb.set_trace()

            if DEBUG:
                logger.info(f'tgt_token:{tgt_token}')
                logger.info(f'aatype_pred:{aatype_pred}')

            if torch.isnan(reduced_loss):
                print(f"logits : {logits[0]}")
                print(f"tgt_token : {tgt_token}")

                print(f"aatype_reduced_loss : {aatype_reduced_loss}")
                print(f"raw_loss: {raw_loss}")

            #
            logging_output = {
                'loss': reduced_loss.detach().cpu().item(),
                'aatype_loss': aatype_reduced_loss.detach().cpu().item(),
                'aatype_acc': aatype_acc.detach().cpu().item(),
                'torsion_loss': reduced_torsion_loss.detach().cpu().item(),
                'torsion_acc': reduced_torsion_acc.detach().cpu().item(),
                'sample_size': sample_size,                      # 本来就是 int / 无梯度
                'tm_aatype_acc': tm_aatype_acc.detach().cpu().item(),
                'tm_aatype_loss': tm_aatype_loss.detach().cpu().item(),
                'nontm_aatype_acc': nontm_aatype_acc.detach().cpu().item(),
                'nontm_aatype_loss': nontm_aatype_loss.detach().cpu().item(),
                'ntokens': sample_size,                          # int
                'nsentences': batch_size,                        # int
                'weight':weight.float().mean().item(),
                'wy_recall': wy_recall.detach().cpu().item(),
                'gate_mean': sample_cfgs.get('gate').mean().item() if sample_cfgs.get('gate') is not None else None,
            }
            #使得reduced——loss张量的数据类型和logits一致
            reduced_loss = reduced_loss.type_as(logits)
            if os.environ.get('DEBUG_GNORM'):
                def _make_hook(n):
                    def _hook(grad):
                        import sys
                        msg = f'DEBUG_GNORM: {n} grad=None' if grad is None else f'DEBUG_GNORM: {n} grad_norm={grad.norm().item():.10f}'
                        print(msg, file=sys.stderr, flush=True)
                        return grad
                    return _hook
                for _n, _p in model.named_parameters():
                    if 'mem_logit' in _n:
                        _p.register_hook(_make_hook(_n))
            # import ipdb; ipdb.set_trace()


            #返回损失函数，损失函数字典，以及sample的size
            return reduced_loss, sample_size, logging_output


    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) :
        """Aggregate logging outputs from data parallel training."""

        #这个函数是每个batch调用一次，例如，batchsize=4,梯度累计=16，那么就每64个样算一遍
        node_num = len(logging_outputs)#16=梯度累计
        if node_num == 0:
            node_num = 1
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("sample_size", sample_size, sample_size, round=4)

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / node_num, 1, round=4)

        aatype_loss_sum = sum(log.get("aatype_loss", 0) for log in logging_outputs)
        metrics.log_scalar("aatype_loss", aatype_loss_sum / node_num, 1, round=4)
        aatype_acc_sum = sum(log.get("aatype_acc", 0) for log in logging_outputs)
        metrics.log_scalar("aaype_acc", aatype_acc_sum/node_num, 1, round=4)

        torsion_loss_sum = sum(log.get("torsion_loss", 0) for log in logging_outputs)
        metrics.log_scalar("torsion_loss", torsion_loss_sum / node_num, 1, round=4)
        torsion_acc_sum = sum(log.get("torsion_acc", 0) for log in logging_outputs)
        metrics.log_scalar("torsion_acc", torsion_acc_sum/node_num, 1, round=4)

        torsion_loss_sum = sum(log.get("tm_aatype_acc", 0) for log in logging_outputs)
        metrics.log_scalar("tm_aatype_acc", torsion_loss_sum / node_num, 1, round=4)
        torsion_acc_sum = sum(log.get("tm_aatype_loss", 0) for log in logging_outputs)
        metrics.log_scalar("tm_aatype_loss", torsion_acc_sum/node_num, 1, round=4)
        torsion_loss_sum = sum(log.get("nontm_aatype_acc", 0) for log in logging_outputs)
        metrics.log_scalar("nontm_aatype_acc", torsion_loss_sum / node_num, 1, round=4)
        torsion_acc_sum = sum(log.get("nontm_aatype_loss", 0) for log in logging_outputs)
        metrics.log_scalar("nontm_aatype_loss", torsion_acc_sum/node_num, 1, round=4)
        torsion_acc_sum = sum(log.get("weight", 0) for log in logging_outputs)
        metrics.log_scalar("weight", torsion_acc_sum/node_num, 1, round=4)
        wy_recall_sum = sum(log.get("wy_recall", 0) for log in logging_outputs)
        metrics.log_scalar("wy_recall", wy_recall_sum/node_num, 1, round=4)
        gate_vals = [log.get("gate_mean") for log in logging_outputs if log.get("gate_mean") is not None]
        if gate_vals:
            metrics.log_scalar("gate_mean", sum(gate_vals)/len(gate_vals), 1, round=4)





    @staticmethod
    def logging_outputs_can_be_summed():
        return False


if __name__ == "__main__":
    char2weight = {k: 1.2 if k in {'H','B','I','C','L'} else 1.0 for k in region_dict.keys()}
    # 2. 把映射表转成 tensor，方便一次性索引，假设字符的整数编码就是 region_dict 的值（0~9）
    weight_table = torch.tensor(
            [char2weight[reverse_region_dict[i]] for i in range(len(region_dict))],
            dtype=torch.float32
        )
    print(weight_table)
    pdbtm_regions_mask = torch.tensor([2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,4])
    region_weight = weight_table[pdbtm_regions_mask]  # (B, L) #基于regions的权重，可以直接拿来用
    print(region_weight)
