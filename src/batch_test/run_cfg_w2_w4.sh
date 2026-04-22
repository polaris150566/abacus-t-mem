#!/usr/bin/env bash
source /home/chenty/miniconda3/etc/profile.d/conda.sh
conda activate abacust
set -euo pipefail

BATCH_TEST_DIR="/home/chenty/abacust_mem/src/batch_test"
DATA_ROOT="/home/chenty/public_data/tmpdb_after_202505"
SEQS_ROOT="${DATA_ROOT}/abacust_design_result/original_output/seqs"
NPY_DIR="${DATA_ROOT}/abacust_design_result/original_output/npys"
TARGET_ID_LIST="${DATA_ROOT}/abacust_design_result/original_output/run_lists/target_ids.txt"

CKPT="/home/chenty/abacust_mem/src/experiments/abacust_mem_depth_absz_single_rbf_concat_noise_with_0124_cluster_dict_b64_cfg/checkpoint/checkpoint100.pt"
TM_RAW="depth_absz_single_rbf_concat_noise_with_0124_cluster_dict_b64_cfg"
MEM_CONFIG="/home/chenty/abacust_mem/src/scripts/configs/mem_config.yaml"

BATCH_SIZE=10
TEMPERATURE=0.1
ITER_NUM=20

echo "=== w=2.0 on GPU 2,3 ==="
python "${BATCH_TEST_DIR}/batch_sample.py" \
    --input-path "${DATA_ROOT}/all_assembled" \
    --design-seq-dir "${SEQS_ROOT}" \
    --npy-dir "${NPY_DIR}" \
    --data-names-file "${TARGET_ID_LIST}" \
    --checkpoint "${CKPT}" \
    --tm-raw "${TM_RAW}" \
    --device-list "2,3" \
    --batchsize "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --iter-num "${ITER_NUM}" \
    --mem-config "${MEM_CONFIG}" \
    --cfg-guidance-scale 2.0 &
PID_W2=$!

echo "=== w=4.0 on GPU 4,5 ==="
python "${BATCH_TEST_DIR}/batch_sample.py" \
    --input-path "${DATA_ROOT}/all_assembled" \
    --design-seq-dir "${SEQS_ROOT}" \
    --npy-dir "${NPY_DIR}" \
    --data-names-file "${TARGET_ID_LIST}" \
    --checkpoint "${CKPT}" \
    --tm-raw "${TM_RAW}" \
    --device-list "4,5" \
    --batchsize "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --iter-num "${ITER_NUM}" \
    --mem-config "${MEM_CONFIG}" \
    --cfg-guidance-scale 4.0 &
PID_W4=$!

echo "Launched: w=2.0 (PID $PID_W2, GPU 2,3), w=4.0 (PID $PID_W4, GPU 4,5)"
wait $PID_W2 $PID_W4
echo "[$(date)] Both finished."
