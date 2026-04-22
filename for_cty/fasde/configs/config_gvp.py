import ml_collections as mlc
import copy
from typing import Any


N_RES = "number of residues"
N_MSA = "number of MSA sequences"
N_EXTRA_MSA = "number of extra MSA sequences"
N_TPL = "number of templates"


d_pair = mlc.FieldReference(128, field_type=int)
d_msa = mlc.FieldReference(256, field_type=int)
d_template = mlc.FieldReference(64, field_type=int)
d_extra_msa = mlc.FieldReference(64, field_type=int)
d_single = mlc.FieldReference(384, field_type=int)
max_recycling_iters = mlc.FieldReference(3, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(200, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(3e4, field_type=float)
use_templates = mlc.FieldReference(True, field_type=bool)
is_multimer = mlc.FieldReference(False, field_type=bool)

c_beta_min = mlc.FieldReference(0.1, field_type=float)
c_beta_max = mlc.FieldReference(20.0, field_type=float)
r_sigma_min = mlc.FieldReference(0.01, field_type=float)
r_sigma_max = mlc.FieldReference(5.0, field_type=float)
chi_sigma_min = mlc.FieldReference(0.01, field_type=float)
chi_sigma_max = mlc.FieldReference(5.0, field_type=float)
downsample_scale = mlc.FieldReference(1, field_type=int)

use_clamped_fape_prob = mlc.FieldReference(0.9, field_type=float)

trans_scale_factor = mlc.FieldReference(10.0, field_type=float)

def base_config():
    return mlc.ConfigDict(
        {
            "data": {
                "globals": {
                    "trans_scale_factor": trans_scale_factor,
                    # "use_prior_prob": 1.0,
                    "freeze_receptor": False,
                    "use_clamped_fape_prob": use_clamped_fape_prob,
                    "downsample_scale": downsample_scale,
                    "use_esm": False,
                },

                "full_atom": {
                    "fix_receptor_backbone": True,
                    "use_esm": False,
                }
            },
            "globals": {
                "chunk_size": chunk_size,
                "block_size": None,
                "d_pair": d_pair,
                "d_msa": d_msa,
                "d_template": d_template,
                "d_extra_msa": d_extra_msa,
                "d_single": d_single,
                "eps": eps,
                "inf": inf,
                "max_recycling_iters": max_recycling_iters,
                "alphafold_original_mode": False,
                "use_esm": False,
                "esm_dim": 1280,
                "num_aatypes": 22,
                "max_distance": 20.0,
                "dist_embed_dim": 200,
            },
            "model": {
                "encoder":{
                    "gvp_feat":{
                        "top_k_neighbors": 30,
                        "node_hidden_dim_scalar":1024,
                        "node_hidden_dim_vector":256,
                        "edge_hidden_dim_scalar":32,
                        "edge_hidden_dim_vector":1,
                        "dropout":0.1,
                        "num_encoder_layers":4,
                        "esm_dim":1280,
                        "num_aatype":22,
                        "num_lig_type":199,
                        "n_rbf_bins":100,
                        "num_positional_embeddings":16,
                        "t_embed_dim":32,
                        "attr_dim":6,
                    },
                    "embed_dim":256,
                    "encoder_layers":6,
                    "gvp_arch":"vt_medium_with_invariant_gvp",
                    "node_hidden_dim_scalar":1024,
                    "node_hidden_dim_vector":256,
                    "dropout":0.1,
                    "encoder_ffn_embed_dim":1024,
                    "encoder_attention_heads":4,
                    "attention_dropout":0.0,
                    "bias_weight":-1.0,
                },
            },
            "loss": {
                "weights": {
                    "score_matching": 1.0,
                    "lig_bond": 0.,
                    "lig_angle": 0.,
                    "lig_torsion": 0.,
                    "prot_bond": 0.,
                    "prot_angle": 0.,
                    "prot_torsion": 0.,
                    "prot_phipsi": 0.,
                    "prot_sidechains": 0.,
                },

                # "weights": {
                #     "score_matching": 1.0,
                #     "lig_bond": 0.3,
                #     "lig_angle": 0.3,
                #     "lig_torsion": 0.3,
                #     "prot_bond": 0.3,
                #     "prot_angle": 0.3,
                #     "prot_torsion": 0.3,
                #     "prot_phipsi": 0.3,
                #     "prot_sidechains": 0.3,
                # },
            },
        }
    )


def recursive_set(c: mlc.ConfigDict, key: str, value: Any, ignore: str = None):
    with c.unlocked():
        for k, v in c.items():
            if ignore is not None and k == ignore:
                continue
            if isinstance(v, mlc.ConfigDict):
                recursive_set(v, key, value)
            elif k == key:
                c[k] = value


def model_config(args, train=False):
    c = copy.deepcopy(base_config())

    if getattr(args, 'freeze_receptor', False):
        c.model.prior.use_prior_prob = -1.0
        c.data.globals.freeze_receptor = True
        c.loss.sde.rec_rot_weight = 0.0
        c.loss.sde.rec_tr_weight = 0.0
        c.loss.sde.rec_tor_weight = 0.0
        c.loss.distogram.weight = 0.0
        c.loss.fape.weight = 0.0
        c.loss.supervised_chi.weight = 0.0
        c.loss.prior_distogram.weight = 0.0

    # if 'af2_init' in name:
    if getattr(args, 'af2_init', False):
        c.model.evoformer_stack.num_blocks = 48
        c.model.structure_module.num_blocks = 8
        c.model.downsampler.scale = 1
        c.model.upsampler.scale = 1
        c.loss.violation.weight = 0.02

    if train:
        c.globals.chunk_size = None
    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c
