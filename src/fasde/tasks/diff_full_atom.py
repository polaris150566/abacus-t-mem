# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
import torch
from argparse import Namespace
# from ..configs.config_gvp import base_config


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir= './log_dir/')

import sys

sys.path.append('/home/chenty/abacust_mem/src/fasde/data')
sys.path.append('/home/chenty/abacust_mem/src/fasde')
sys.path.append('/home/chenty/abacust_mem/src')
from ..data.fullatom_dataset import  FullAtomDataset, MixedPDBAFDBDataset
from fairseq.tasks import  register_task,LegacyFairseqTask
from fairseq import utils
from fairseq import criterions


from collections import namedtuple

logger = logging.getLogger(__name__)

Config = namedtuple('Config', ['reuse_dataloader'])


@register_task("diff_full_atom")
class DiffFullAtomTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        # parser.add_argument("data", help="manifest root path")
        #这个data的参数是哪里来的？，从来没见过啊
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-protein-sequence-len",
            default=384,
            type=int,
            help="maximum protein sequence len",
        )
        parser.add_argument(
            "--pdbbind-path",
            default="/train14/superbrain/lhchen/data/alphafold_database/data",
            type=str,
            help="AFDB data path",
        )
        parser.add_argument(
            "--pdbtm_file_path",
            default="/home/chenty/abacust_mem/src/data_utils/prot_lists/pdbtm_data/tm_npys/",
            type=str,
            help="pdbtm data path",
        )
        parser.add_argument(
            "--pdb-path",
            default="/home/chenty/abacust_mem/src/data/pdbs",
            type=str,
            help="CATH data path",
        )
        parser.add_argument(
            "--npy-path",
            default = '/home/chenty/abacust_mem/src/data_utils/data_process/data_storage/npys/all_npy',
            type=str,
            help="npy data path",
        )
        parser.add_argument(#标记CA原子的半径
            "--ca-radius",
            default=9.0,
            type=float,
            help="ca radius",
        )
        parser.add_argument(#action表示只要出现--embed_unimol_reprs参数则记为true（实际上是有的）
            "--embed_unimol_reprs",
            action='store_true',
            default=False
        )
        parser.add_argument("--afdb", action="store_true", default=False)
        parser.add_argument("--afdb_npy_path", default="")
        parser.add_argument("--afdb_list_path", default="")
        parser.add_argument("--afdb_tmdet_npy_path", default="")
        parser.add_argument("--pdb_ratio", type=float, default=1)
        parser.add_argument("--afdb_apply_noise_prob", type=float, default=1)
        parser.add_argument("--afdb_noise_sigma", type=float, default=0.2)

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.cfg = Config(reuse_dataloader=False)
        #这里config命名可以一个变量，表示是否重新使用dataloader


    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def build_criterion(self, args):
        return criterions.build_criterion(args, self)

    def _load_data_path(self, epoch):
        return '/home/chenty/abacust_mem/src/fasde/data/pdbs/cluster_center.txt'
    #这个是聚类中心，注意这里在划分训练集的时候所有的cluster都在，但是训练集不是，若想验证需要重新写代码

    def load_dataset(self, split, epoch=1,  **kwargs):
        """fairseq会在构建任务的过程中直接调用这个函数，

        参数包括训练类型（如果是valid则epoch直接填1，combine = true）
        如果不是验证，则协商数据集的性质，以及epoch是自己的epoch，combine = false"""

        if not self.args.afdb:

            data_path = self._load_data_path(1 if split == 'valid' else epoch)#看一看split是啥,如果是valid的话输入1，如果不是的话输入当前轮数

            logger.info(f'loading data shard: {data_path} ... ')

            self.datasets[split] = FullAtomDataset(#这个字典在 LegacyFairseqTask 的 __init__ 方法中被初始化为一个空字典。因此，当在子类 DiffFullAtomTask 中访问 self.datasets 时，它已经被定义好了
                self.args.seed,
                data_path,
                self.args.pdbtm_file_path,
                self.args.pdb_path,
                self.args.npy_path,
                split=split,
                mode=split,
                crop_size = self.args.max_protein_sequence_len,
                resize_len = False,
                lig_single_dim = 199,
                lig_pair_dim = 4,
                num_aatypes = 22,
                fix_receptor_backbone = True,
                ca_radius = self.args.ca_radius,
                lig_neighbor_seq_mask=self.args.lig_neighbor_seq_mask,
                embed_unimol_reprs=self.args.embed_unimol_reprs
            )
        elif self.args.afdb:
            data_path = self._load_data_path(1 if split == 'valid' else epoch)#看一看split是啥,如果是valid的话输入1，如果不是的话输入当前轮数

            logger.info(f'loading data shard: {data_path} ... ')

            self.datasets[split] = MixedPDBAFDBDataset(
                self.args.seed,
                data_path,
                self.args.pdbtm_file_path,
                self.args.pdb_path,
                self.args.npy_path,
                split=split,
                mode=split,
                crop_size = self.args.max_protein_sequence_len,
                resize_len = False,
                lig_single_dim = 199,
                lig_pair_dim = 4,
                num_aatypes = 22,
                fix_receptor_backbone = True,
                ca_radius = self.args.ca_radius,
                lig_neighbor_seq_mask=self.args.lig_neighbor_seq_mask,
                embed_unimol_reprs=self.args.embed_unimol_reprs,

                afdb_npy_path = self.args.afdb_npy_path, afdb_list_path = self.args.afdb_list_path, afdb_tmdet_npy_path = self.args.afdb_tmdet_npy_path,
                afdb_plddt_threshold = 70.0, afdb_ptm_threshold = 0.5,
                pdb_ratio = self.args.pdb_ratio,                     # PDB 采样比例 (0-1)
                afdb_noise_sigma = self.args.afdb_noise_sigma,              # AFDB 骨架噪声水平
                afdb_apply_noise_prob = self.args.afdb_apply_noise_prob,         # 对 AFDB 应用噪声的概率
            )



    def build_model(self, args):
        model = super(DiffFullAtomTask, self).build_model(args)
        return model
    #########################################################
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    # def valid_step(self, sample, model, criterion, ema_model = None):
    #     _loss = None
    #     sample_size = None
    #     logging_output = {}

    #     return _loss, sample_size, logging_output


    # _loss, sample_size, logging_output = self.task.valid_step(
    #                 sample, self.model, self.criterion, **extra_kwargs
    #             )
    ##################################################################


    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
