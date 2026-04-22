#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash run_single.sh /abs/path/to/target.pdb [root_dir]

Description:
  Run ABACUST membrane inference for a single PDB.
  Outputs are written under <pdb_dir>/<pdb_stem>/:
    - npys/
    - designresult/

Arguments:
  pdb_path   Absolute or relative path to a single PDB file.
  root_dir   Optional directory containing the pdbtm json file used by
             seq_design_with_lig_raw_pdb.py. If omitted, the script tries:
             1. <pdb_dir>/<pdb_stem>/pdbtm
             2. <pdb_dir>/<pdb_stem>.json  (it will symlink this into pdbtm/)
             3. <script_dir>/pdbs

Environment overrides:
  DEVICE, BATCHSIZE, CKPT, ITER_NUM, TEMPERATURE, SUFFIX,
  PROTEIN_CHAIN, PLM_PARAM_DIR, SELFCOND_PLM, MAX_LIG_NUM, AUGMENT_EPS,
  MASK_MODE
USAGE
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

pdb_input=$1
if ! pdb_path=$(realpath "$pdb_input" 2>/dev/null); then
    echo "Error: failed to resolve pdb path: $pdb_input" >&2
    exit 1
fi
if [ ! -f "$pdb_path" ]; then
    echo "Error: pdb file not found: $pdb_path" >&2
    exit 1
fi

pdb_dir=$(dirname "$pdb_path")
pdb_name=$(basename "$pdb_path")
pdb_stem=${pdb_name%.*}
out_root="$pdb_dir/$pdb_stem"
npydir="$out_root/npys"
save_dir="$out_root/designresult"
pdb_list="$out_root/pdb_list.txt"
auto_root_dir="$out_root/pdbtm"

batchsize=${BATCHSIZE:-10}
device=${DEVICE:-3}
ckpt=${CKPT:-$repo_root/src/experiments/abacust_mem_raw/checkpoint/checkpoint_best_noise650M.pt}
iter_num=${ITER_NUM:-10}
temperature=${TEMPERATURE:-0.1}
suffix=${SUFFIX:-single}
protein_chain=${PROTEIN_CHAIN:-A}
plm_param_dir=${PLM_PARAM_DIR:-/database/lyf_database/pretrain_lm/esm/param}
selfcond_plm=${SELFCOND_PLM:-650M}
max_lig_num=${MAX_LIG_NUM:-3}
augment_eps=${AUGMENT_EPS:-0.2}
mask_mode=${MASK_MODE:-aatype_nll}

mkdir -p "$npydir" "$save_dir"
printf '%s\n' "$pdb_path" > "$pdb_list"
find "$npydir" -type f -name '*.npy' -delete

if [ $# -eq 2 ]; then
    if ! root_dir=$(realpath "$2" 2>/dev/null); then
        echo "Error: failed to resolve root_dir: $2" >&2
        exit 1
    fi
else
    if [ -d "$auto_root_dir" ] && find "$auto_root_dir" -maxdepth 1 -type f -name '*.json' | grep -q .; then
        root_dir="$auto_root_dir"
    elif [ -f "$pdb_dir/$pdb_stem.json" ]; then
        mkdir -p "$auto_root_dir"
        ln -sfn "$pdb_dir/$pdb_stem.json" "$auto_root_dir/$pdb_stem.json"
        root_dir="$auto_root_dir"
    elif [ -d "$script_dir/pdbs" ] && find "$script_dir/pdbs" -maxdepth 1 -type f -name '*.json' | grep -q .; then
        root_dir="$script_dir/pdbs"
    else
        echo "Error: no valid root_dir with a pdbtm json file was found." >&2
        echo "Provide it as the second argument, or place one of these before rerunning:" >&2
        echo "  1. $out_root/pdbtm/*.json" >&2
        echo "  2. $pdb_dir/$pdb_stem.json" >&2
        echo "  3. $script_dir/pdbs/*.json" >&2
        exit 1
    fi
fi

if [ ! -d "$root_dir" ]; then
    echo "Error: root_dir is not a directory: $root_dir" >&2
    exit 1
fi

if ! find "$root_dir" -maxdepth 1 -type f -name '*.json' | grep -q .; then
    echo "Error: no .json file found in root_dir: $root_dir" >&2
    exit 1
fi

if [ ! -f "$ckpt" ]; then
    echo "Error: checkpoint not found: $ckpt" >&2
    exit 1
fi

echo "PDB: $pdb_path"
echo "Output root: $out_root"
echo "Npy dir: $npydir"
echo "Design result dir: $save_dir"
echo "root_dir: $root_dir"
echo "device: $device"
echo "batchsize: $batchsize"

echo "Running structure preprocess"
python3 "$repo_root/src/data/data_utils/make_feature_from_pdb.py" \
    --pdb_list "$pdb_list" \
    --out_dir "$npydir" \
    --protein_chain "$protein_chain" \
    --mode inference

echo "Running design"
CUDA_VISIBLE_DEVICES="$device" python3 "$repo_root/src/inference/seq_design_with_lig_raw_pdb.py" \
    --npy_dir "$npydir/all_npy" \
    --save_dir "$save_dir" \
    --iter_num "$iter_num" \
    --temperature "$temperature" \
    --suffix "$suffix" \
    --checkpoint "$ckpt" \
    --consider_lig \
    --embed_unimol_reprs \
    --nar \
    --esm_refinement \
    --root_dir "$root_dir" \
    --batchsize "$batchsize" \
    --PLM_selfcond 1 \
    --PLM_param_dir "$plm_param_dir" \
    --selfcondPLM "$selfcond_plm" \
    --augment_eps "$augment_eps" \
    --mask_mode "$mask_mode" \
    --write_allatom_model \
    --tm_raw raw \
    --max_lig_num "$max_lig_num"
