#!/usr/bin/env bash
source /home/chenty/miniconda3/etc/profile.d/conda.sh
conda activate abacust
set -euo pipefail

BATCH_TEST_DIR="/home/chenty/abacust_mem/src/batch_test"
PREPARE_LIST_SCRIPT="${BATCH_TEST_DIR}/prepare_tmpdb_run_list.py"
FEATURE_SCRIPT="/home/chenty/abacust_mem/src/data/data_utils/make_feature_from_pdb.py"

DATA_ROOT="/home/chenty/public_data/tmpdb_after_202505"
ALL_PDB_DIR="${DATA_ROOT}/all_pdbs"
CLUSTER_DICT="${DATA_ROOT}/data_storage/cluster_dict.npy"
CLUSTER_CENTERS_TXT="${DATA_ROOT}/data_storage/cluster_centers.txt"

SELECTION_MODE="values"
LIMIT=""

OUT_ROOT="${DATA_ROOT}/abacust_design_result/original_output"
RUN_LIST_DIR="${OUT_ROOT}/run_lists"
FEATURE_OUT_DIR="${OUT_ROOT}/npys"
SEQ_OUT_DIR="${OUT_ROOT}/seqs"

PDB_PATH_LIST="${RUN_LIST_DIR}/pdb_paths.txt"
TARGET_ID_LIST="${RUN_LIST_DIR}/target_ids.txt"

CHECKPOINTS=(
  /home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint80.pt
  /home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint90.pt
  /home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint100.pt
)

TM_RAWS=(
  zero_with_0124_cluster_dict_b64_lr25
  zero_with_0124_cluster_dict_b64_lr25
  zero_with_0124_cluster_dict_b64_lr25
)

DEVICE_LISTS=(
  1
  4
  6
)

MEM_CONFIGS=(
  ""
  ""
  ""
)

CFG_SCALES=(
  1.0
  1.0
  1.0
)

BATCH_SIZE=10
TEMPERATURE=0.1
ITER_NUM=20
FEATURE_NUM_WORKERS=32

prepare_dirs() {
  mkdir -p "${RUN_LIST_DIR}"
  mkdir -p "${FEATURE_OUT_DIR}"
  mkdir -p "${SEQ_OUT_DIR}"
}

check_run_arrays() {
  local count
  count="${#CHECKPOINTS[@]}"
  if [ "${#TM_RAWS[@]}" -ne "${count}" ] || [ "${#DEVICE_LISTS[@]}" -ne "${count}" ]; then
    echo "run config arrays must have the same length" >&2
    exit 1
  fi
}

build_run_args() {
  local index
  RUN_ARGS=()
  for ((index = 0; index < ${#CHECKPOINTS[@]}; index++)); do
    RUN_ARGS+=(--checkpoint "${CHECKPOINTS[index]}")
    RUN_ARGS+=(--tm-raw "${TM_RAWS[index]}")
    RUN_ARGS+=(--device-list "${DEVICE_LISTS[index]}")
    if [ -n "${MEM_CONFIGS[index]:-}" ]; then
      RUN_ARGS+=(--mem-config "${MEM_CONFIGS[index]}")
    fi
    if [ -n "${CFG_SCALES[index]:-}" ]; then
      RUN_ARGS+=(--cfg-guidance-scale "${CFG_SCALES[index]}")
    fi
  done
}

print_run_configs() {
  local index
  for ((index = 0; index < ${#CHECKPOINTS[@]}; index++)); do
    echo "[run] checkpoint: ${CHECKPOINTS[index]}"
    echo "[run] tm_raw: ${TM_RAWS[index]}"
    echo "[run] device_list: ${DEVICE_LISTS[index]}"
  done
}

prepare_lists() {
  prepare_dirs

  if [ -n "${LIMIT}" ]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
  else
    LIMIT_ARGS=()
  fi

  python "${PREPARE_LIST_SCRIPT}" \
    --all-pdb-dir "${ALL_PDB_DIR}" \
    --cluster-dict "${CLUSTER_DICT}" \
    --cluster-centers-txt "${CLUSTER_CENTERS_TXT}" \
    --selection-mode "${SELECTION_MODE}" \
    --pdb-path-list "${PDB_PATH_LIST}" \
    --target-id-list "${TARGET_ID_LIST}" \
    "${LIMIT_ARGS[@]}"
}

run_features() {
  echo "[feature] start"
  echo "[feature] pdb list: ${PDB_PATH_LIST}"
  echo "[feature] output dir: ${FEATURE_OUT_DIR}"
  echo "[feature] workers: ${FEATURE_NUM_WORKERS}"

  python "${FEATURE_SCRIPT}" \
    --pdb_list "${PDB_PATH_LIST}" \
    --out_dir "${FEATURE_OUT_DIR}" \
    --mode inference \
    --num_workers "${FEATURE_NUM_WORKERS}" \
    --skip_existing
}

run_prepare_step() {
  prepare_lists
}

run_feature_step() {
  run_features
}

run_sample_step() {
  check_run_arrays
  build_run_args

  echo "[sample] start"
  print_run_configs

  python "${BATCH_TEST_DIR}/batch_sample.py" \
    --input-path "${ALL_PDB_DIR}" \
    --design-seq-dir "${SEQ_OUT_DIR}" \
    --npy-dir "${FEATURE_OUT_DIR}" \
    --data-names-file "${TARGET_ID_LIST}" \
    --batchsize "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --iter-num "${ITER_NUM}" \
    "${RUN_ARGS[@]}"
}

run_gather_step() {
  local index
  local ckpt_stem
  local root_dir

  check_run_arrays

  echo "[gather] start"
  print_run_configs

  for ((index = 0; index < ${#CHECKPOINTS[@]}; index++)); do
    ckpt_stem="$(basename "${CHECKPOINTS[index]}" .pt)"
    root_dir="${SEQ_OUT_DIR}/T_${TEMPERATURE}_R_${ITER_NUM}_esm_refined_${TM_RAWS[index]}/${ckpt_stem}"

    python "${BATCH_TEST_DIR}/batch_gather_result.py" \
      --root-dir "${root_dir}" \
      --target-list-file "${TARGET_ID_LIST}" \
      --batch-size "${BATCH_SIZE}" \
      --iter-number "${ITER_NUM}" \
      --mode all
  done
}

main() {
  # run_prepare_step
  # run_feature_step
  run_sample_step
  # run_gather_step
}

main
