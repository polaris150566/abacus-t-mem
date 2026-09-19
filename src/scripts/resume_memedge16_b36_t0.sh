#!/bin/bash
# ------------- memedge16 b36 重训: 有效batch 3x12=36 + valid时间步固定0 + mem_edge16架构 
GPU=${GPU:-0}
tm_raw="zero_afdb_with_0124afdb_cluster_dict_1_3_b36_t0_memedge16"
CKPT='/home/chenty/abacust_mem/src/experiments/abacust_mem_zero/checkpoints/checkpoint_best_noise650M.pt'
SAVE_DIR=/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}/checkpoints
LOG_DIR="/home/chenty/abacust_mem/src/scripts/log"
DATA_STORAGE="/home/chenty/public_data/abacust_mem_data/data_storage"
ESM_PRETRAINED="/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

mkdir -p $LOG_DIR $SAVE_DIR

# ------------- 训练超参 (lr25 同款, 仅 batch 改 3x12=36) -------------
LR=5e-5
MAX_EPOCH=100
BATCH_SIZE=3
NUM_WORKERS=4
UPDATE_FREQ=12
LOG_INTERVAL=100
SAVE_INTERVAL=10
VALIDATE_INTERVAL=1
KEEP_LAST_EPOCHS=20
PATIENCE=0
SEED=42

EXP_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}"
EFFECTIVE_BS=$((BATCH_SIZE * UPDATE_FREQ))
{
    echo "lr: $LR"
    echo "effective_batch_size: $EFFECTIVE_BS (3 x 12)"
    echo "valid_timestep: fixed 0 (design from scratch)"
    echo "base_ckpt: $CKPT"
} > "$EXP_DIR/experiment_config.yaml"

cd /home/chenty/abacust_mem/src

run_training() {
      local log_file="${LOG_DIR}/train_${tm_raw}_$(date +%F_%H-%M-%S).log"
      echo "[$(date)] Starting training... log: $log_file"
      CUDA_VISIBLE_DEVICES=$GPU /home/chenty/miniconda3/envs/abacust/bin/fairseq-train \
            --user-dir /home/chenty/abacust_mem/src/fasde \
            --task diff_full_atom --arch diff_full_atom_base --criterion diff_full_atom_criterion \
            --optimizer adam \
            --lr $LR --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 1000 \
            --max-epoch $MAX_EPOCH \
            --batch-size $BATCH_SIZE \
            --required-batch-size-multiple 1 \
            --batch-size-valid 1 \
            --save-dir $SAVE_DIR \
            --num-workers $NUM_WORKERS \
            --update-freq $UPDATE_FREQ \
            --distributed-world-size 1 \
            --log-interval $LOG_INTERVAL \
            --save-interval $SAVE_INTERVAL \
            --validate-interval $VALIDATE_INTERVAL \
            --keep-last-epochs $KEEP_LAST_EPOCHS \
            --patience $PATIENCE \
            --seed $SEED \
            --config-yaml config.yaml \
            --max-protein-sequence-len 256 \
            --pdb-path "$DATA_STORAGE/assembled_pdbs/" \
            --npy-path "$DATA_STORAGE/merged_npys/all_npy" \
            --pdbtm_file_path "$DATA_STORAGE/tmdet_result/npys/" \
            --afdb \
            --pdb_ratio 0.25 \
            --afdb_npy_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/npys/all_npy" \
            --afdb_list_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/Nonredundant_merged_cluster_dict.npy" \
            --afdb_tmdet_npy_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/tm_npys" \
            --afdb_apply_noise_prob 1 \
            --afdb_noise_sigma 0.2 \
            --ca-radius 9.0 \
            --diff_T 40 \
            --max_iter_num 4 \
            --find-unused-parameters \
            --embed_unimol_reprs \
            --pretrained_mpnn_ckpt \
            --pretrained_mpnn_ckpt_f $CKPT \
            --ckpt $CKPT \
            --augment_eps 0.2 \
            --nar \
            --esm_pretrained $ESM_PRETRAINED \
            --tensorboard-logdir $SAVE_DIR/tensorboard \
            --tm_raw $tm_raw \
            --mem_config /home/chenty/abacust_mem/src/scripts/configs/mem_config_memedge16.yaml \
            2>&1 | tee -a "$log_file"
}

while true; do
    run_training
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed with code $EXIT_CODE."
        break
    else
        echo "[$(date)] Training crashed with exit code $EXIT_CODE. Restarting in 20s..."
        sleep 20
    fi
done
