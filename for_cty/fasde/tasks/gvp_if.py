# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from argparse import Namespace
from ..configs.config_gvp import base_config
from ..data.fullatom_dataset import FullAtomDataset
from fairseq.tasks import LegacyFairseqTask, register_task, FairseqTask
from fairseq import utils

from collections import namedtuple

logger = logging.getLogger(__name__)

Config = namedtuple('Config', ['reuse_dataloader'])


@register_task("gvp_if")
class GVPTInverseFoldingTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
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
            "--data-path",
            default="/train14/superbrain/lhchen/protein/full_atom/fullatom_sde_ligand/data/lists",
            type=str,
            help="CATH data path",
        )

        parser.add_argument(
            "--pdb-path",
            default="/train14/superbrain/lhchen/data/PDB/20220102/mmcif",
            type=str,
            help="CATH data path",
        )


    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.cfg = Config(reuse_dataloader=False)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def _load_data_path(self, epoch):
        paths = utils.split_paths(self.args.data)
        data_path = paths[(epoch - 1) % len(paths)]
        return data_path

    def load_dataset(self, split, epoch=1,  **kwargs):
        data_path = self._load_data_path(1 if split == 'valid' else epoch)
        logger.info(f'loading data shard: {data_path} ... ')
        config=base_config()
        self.datasets[split] = FullAtomDataset(
            self.args.seed,
            data_path,
            self.args.pdb_path,
            self.args.pdbbind_path,
            split,
            config=config,
            mode="train",
            crop_size = 128,
            resize_len = False,
            lig_single_dim = 199,
            lig_pair_dim = 4,
            num_aatypes = 22,
            beta_min = 0.1,
            beta_max = 20.0,
            fix_receptor_backbone = True,
        )
        
    def build_model(self, args):
        return super(GVPTInverseFoldingTask, self).build_model(args)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
