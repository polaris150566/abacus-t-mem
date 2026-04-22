# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from argparse import Namespace
from ..configs.config_gvp import base_config
from ..data.fullatom_dataset import FullAtomDataset
from fairseq.tasks import  register_task,LegacyFairseqTask
from fairseq import utils
from fairseq import criterions
from tqdm import tqdm

from collections import namedtuple

logger = logging.getLogger(__name__)

Config = namedtuple('Config', ['reuse_dataloader'])


@register_task("diff_full_atom")
class DiffFullAtomTask(LegacyFairseqTask):
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
        parser.add_argument(
            "--ca-radius",
            default=9.0,
            type=float,
            help="ca radius",
        )
        parser.add_argument(
            "--embed_unimol_reprs", 
            action='store_true', 
            default=False
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.cfg = Config(reuse_dataloader=False)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def build_criterion(self, args):
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

        # dataset = self.datasets['valid_']
        # for data in tqdm(dataset):
        #     pass
        #     # import pdb; pdb.set_trace()
        
    def build_model(self, args):
        return super(DiffFullAtomTask, self).build_model(args)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None
