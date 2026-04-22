import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from ..modules.sequence_designer import ABACUSTDesigner

logger = logging.getLogger(__name__)


@register_model("diff_full_atom")
class DiffFullAtom(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--gvp-arch", type=str) # 'vt_medium_with_invariant_gvp'
        parser.add_argument("--target-shuffle-ratio", type=float)
        parser.add_argument("--nar", action='store_true', default=False)
        parser.add_argument("--lig_neighbor_seq_mask", action='store_true', default=False)
        parser.add_argument("--ligmpnn_init", action='store_true', default=False)
        parser.add_argument("--esm_embedder", action='store_true', default=False)
        parser.add_argument("--max_iter_num", default=4, type=int)
        parser.add_argument("--pre_prot_mode", default='proteinMPNN', type=str) # pifold, proteinMPNN
        parser.add_argument("--merge_mpnn_enc_layer_num", default=3, type=int)
        parser.add_argument("--merge_mpnn_dec_layer_num", default=7, type=int)

        parser.add_argument("--pretrained_mpnn_ckpt", action='store_true', default=False)
        parser.add_argument("--pretrained_mpnn_ckpt_f", type=str, default='')

        parser.add_argument("--esm_pretrained", type=str, default="")
        parser.add_argument("--encode_mpnn", action='store_true', default=False)
        parser.add_argument("--augment_eps", default=0.02, type=float)

        parser.add_argument("--PLM_selfcond", type=str, default='')
        parser.add_argument("--selfcondPLM", type=str, default='')
        # parser.add_argument("--embed_unimol_reprs", type=str, default='')
        parser.add_argument("--PLM_param_dir", type=str, default='')

        parser.add_argument("--diff_T", type=str, default='')
        parser.add_argument("--tm_raw",type=str,default='raw')
        parser.add_argument("--ckpt", type=str, default='')
        parser.add_argument("--mem_config", type=str, default='',
                            help='Path to unified membrane config YAML (e.g. scripts/configs/mem_config.yaml)')




    def __init__(self, args):
        super().__init__()
        self._num_updates = 0

        self.model = ABACUSTDesigner(args)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        model = cls(args)
        return model

    def forward(self, batch):
        output = self.model(
            batch,
        )
        return output

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates


@register_model_architecture("diff_full_atom", "diff_full_atom_base")
def base_architecture(args):
    args.arch = getattr(args, "arch", "vt_medium_with_invariant")
    # args.restore_file = getattr(args, "restore_file", "/home/chenty/abacust_mem/src/experiments/abacust/checkpoint/checkpoint_best_010_650M.pt")