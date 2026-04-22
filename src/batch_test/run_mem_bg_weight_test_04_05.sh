#!/usr/bin/env bash
source /home/chenty/miniconda3/etc/profile.d/conda.sh
conda activate abacust
set -euo pipefail

BATCH_TEST_DIR="/home/chenty/abacust_mem/src/batch_test"
DATA_ROOT="/home/chenty/public_data/tmpdb_after_202505"
SEQS_ROOT="${DATA_ROOT}/abacust_design_result/mem_bg_wy_test/seqs"
NPY_DIR="${DATA_ROOT}/abacust_design_result/original_output/npys"
TARGET_ID_LIST="${DATA_ROOT}/abacust_design_result/original_output/run_lists/target_ids.txt"

CKPT="/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint100.pt"
TM_RAW="zero_with_0124_cluster_dict_b64_lr25"

BATCH_SIZE=10
TEMPERATURE=0.1
ITER_NUM=20

mkdir -p "${SEQS_ROOT}"

python "${BATCH_TEST_DIR}/batch_sample.py" \
    --input-path "${DATA_ROOT}/all_pdbs" \
    --design-seq-dir "${SEQS_ROOT}" \
    --npy-dir "${NPY_DIR}" \
    --data-names-file "${TARGET_ID_LIST}" \
    --checkpoint "${CKPT}" \
    --checkpoint "${CKPT}" \
    --tm-raw "${TM_RAW}" \
    --tm-raw "${TM_RAW}" \
    --device-list "1" \
    --device-list "5" \
    --batchsize "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --iter-num "${ITER_NUM}" \
    --mem-bg-weight 0.4 \
    --mem-bg-weight 0.5
