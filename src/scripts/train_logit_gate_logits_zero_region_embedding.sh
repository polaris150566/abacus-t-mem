#!/bin/bash
set -o pipefail

FD_MONITOR_INTERVAL=60
FD_MONITOR_PID=""

fd_count() {
    local pid="$1"
    if [ -d "/proc/$pid/fd" ]; then
        ls "/proc/$pid/fd" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

start_fd_monitor() {
    local monitor_log="$1"
    (
        while true; do
            local now
            now=$(date '+%F %T')
            local pids
            pids=$(pgrep -u "$USER" -f "fairseq-train.*${tm_raw}" || true)
            if [ -z "$pids" ]; then
                sleep 5
                continue
            fi
            for pid in $pids; do
                [ -d "/proc/$pid" ] || continue
                local parent_fd
                parent_fd=$(fd_count "$pid")
                echo "[$now] role=fairseq pid=$pid fd=$parent_fd" >> "$monitor_log"
                local child
                for child in $(pgrep -P "$pid" || true); do
                    [ -d "/proc/$child" ] || continue
                    echo "[$now] role=child parent=$pid pid=$child fd=$(fd_count "$child")" >> "$monitor_log"
                done
            done
            sleep "$FD_MONITOR_INTERVAL"
        done
    ) &
    FD_MONITOR_PID=$!
}

stop_fd_monitor() {
    if [ -n "$FD_MONITOR_PID" ]; then
        kill "$FD_MONITOR_PID" 2>/dev/null || true
        wait "$FD_MONITOR_PID" 2>/dev/null || true
        FD_MONITOR_PID=""
    fi
}

tm_raw="zero_with_0124_cluster_dict_b64_lr25_logit_gate_logits_zero_region_embedding"
CKPT="/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoint/checkpoint70.pt"
MEM_CONFIG="/home/chenty/abacust_mem/src/scripts/configs/mem_config_logit_gate_decoder_pre_unified_zero_region_embedding.yaml"
SAVE_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}/checkpoints"
LOG_DIR="/home/chenty/abacust_mem/src/scripts/log"
GPU_ID="${GPU_ID:-2}"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"
chmod 755 "$SAVE_DIR"

if [ ! -f "$CKPT" ]; then
    echo "Base checkpoint not found: $CKPT" >&2
    exit 1
fi

if [ ! -f "$MEM_CONFIG" ]; then
    echo "Membrane config not found: $MEM_CONFIG" >&2
    exit 1
fi

LR=2.5e-4
MAX_EPOCH=100
BATCH_SIZE=8
NUM_WORKERS=4
UPDATE_FREQ=8
LOG_INTERVAL=100
SAVE_INTERVAL=10
VALIDATE_INTERVAL=1
KEEP_LAST_EPOCHS=10
PATIENCE=0
SEED=42

MAX_PROTEIN_SEQUENCE_LEN=256
PDB_PATH="/home/chenty/public_data/abacust_mem_data/data_storage/assembled_pdbs/"
PDBTM_PATH="/home/chenty/public_data/abacust_mem_data/data_storage/tmdet_result/npys/"
NPY_PATH="/home/chenty/public_data/abacust_mem_data/data_storage/merged_npys/all_npy"
ESM_PRETRAINED="/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

DIFF_T=40
MAX_ITER_NUM=4
CA_RADIUS=9.0

EXP_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}"
mkdir -p "$EXP_DIR"
EFFECTIVE_BS=$((BATCH_SIZE * UPDATE_FREQ))
{
    echo "tm_raw: $tm_raw"
    echo "gpu_id: $GPU_ID"
    echo "lr: $LR"
    echo "effective_batch_size: $EFFECTIVE_BS"
    echo "base_ckpt: $CKPT"
    echo "mem_config: $MEM_CONFIG"
    echo ""
    cat "$MEM_CONFIG"
} > "$EXP_DIR/experiment_config.yaml"

run_training() {
    local fd_log="${LOG_DIR}/fd_${tm_raw}_$(date +%F_%H-%M-%S).log"
    local log_file="${LOG_DIR}/train_${tm_raw}_$(date +%F_%H-%M-%S).log"
    echo "[$(date)] Starting training... log: $log_file"
    echo "[$(date)] FD monitor log: $fd_log"
    echo "[$(date)] GPU_ID: $GPU_ID"
    start_fd_monitor "$fd_log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" /home/chenty/miniconda3/envs/abacust/bin/fairseq-train \
        --user-dir /home/chenty/abacust_mem/src/fasde \
        --task diff_full_atom --arch diff_full_atom_base --criterion diff_full_atom_criterion \
        --optimizer adam \
        --lr "$LR" --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 1000 \
        --max-epoch "$MAX_EPOCH" \
        --batch-size "$BATCH_SIZE" \
        --batch-size-valid 1 \
        --save-dir "$SAVE_DIR" \
        --num-workers "$NUM_WORKERS" \
        --update-freq "$UPDATE_FREQ" \
        --distributed-world-size 1 \
        --log-interval "$LOG_INTERVAL" \
        --save-interval "$SAVE_INTERVAL" \
        --validate-interval "$VALIDATE_INTERVAL" \
        --keep-last-epochs "$KEEP_LAST_EPOCHS" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --config-yaml config.yaml \
        --max-protein-sequence-len "$MAX_PROTEIN_SEQUENCE_LEN" \
        --pdb-path "$PDB_PATH" \
        --npy-path "$NPY_PATH" \
        --distributed-backend nccl \
        --pdbtm_file_path "$PDBTM_PATH" \
        --ca-radius "$CA_RADIUS" \
        --diff_T "$DIFF_T" \
        --max_iter_num "$MAX_ITER_NUM" \
        --find-unused-parameters \
        --embed_unimol_reprs \
        --pretrained_mpnn_ckpt \
        --pretrained_mpnn_ckpt_f "$CKPT" \
        --ckpt "$CKPT" \
        --tensorboard-logdir "$SAVE_DIR/tensorboard" \
        --tm_raw "$tm_raw" \
        --augment_eps 0.2 \
        --nar \
        --esm_pretrained "$ESM_PRETRAINED" \
        --mem_config "$MEM_CONFIG" \
        2>&1 | tee -a "$log_file"
    local ret=$?
    stop_fd_monitor
    return $ret
}

while true; do
    run_training
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Training completed."
        break
    else
        echo "[$(date)] Crashed (exit $EXIT_CODE). Restarting in 20s..."
        sleep 20
    fi
done
