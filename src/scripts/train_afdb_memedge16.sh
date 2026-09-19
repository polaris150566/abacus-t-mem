#!/bin/bash
# ------------- mem_edge16 微调: base650M + afdb0124afdb(1_3) + mem_edge concat16 + freeze all_except -------------
# 基于 scripts/train_for_afdb.sh, 仅改: tm_raw/存储目录, 加 --mem_config, 指定单卡GPU. 数据集/超参与 afdb 完全一致.
DATA_DIR="/home/chenty/abacust_mem/src/fasde/data"
tm_raw="zero_afdb_with_0124afdb_cluster_dict_1_3_memedge16"
CKPT='/home/chenty/abacust_mem/src/experiments/abacust_mem_zero/checkpoints/checkpoint_best_noise650M.pt'
SAVE_DIR=/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}/checkpoints
LOG_DIR="/home/chenty/abacust_mem/src/scripts/log"
MEM_CONFIG="/home/chenty/abacust_mem/src/scripts/configs/mem_config_memedge16.yaml"

mkdir -p $LOG_DIR
mkdir -p $SAVE_DIR
chmod 755 $SAVE_DIR

# ------------- 训练超参 (与 afdb 一致, 有效batch=4*16=64) -------------
LR=5e-5
MAX_EPOCH=2000
BATCH_SIZE=4
NUM_WORKERS=4
UPDATE_FREQ=16
LOG_INTERVAL=100
SAVE_INTERVAL=1
VALIDATE_INTERVAL=1
KEEP_LAST_EPOCHS=20
PATIENCE=5
SEED=42

CONFIG_YAML="config.yaml"
MAX_PROTEIN_SEQUENCE_LEN=256
PDB_PATH="/home/chenty/public_data/abacust_mem_data/data_storage/assembled_pdbs/"
PDBTM_PATH='/home/chenty/public_data/abacust_mem_data/data_storage/tmdet_result/npys/'
NPY_PATH='/home/chenty/public_data/abacust_mem_data/data_storage/merged_npys/all_npy'

DIFF_T=40
MAX_ITER_NUM=4
CA_RADIUS=9.0

ESM_PRETRAINED="/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

EXP_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}"
EFFECTIVE_BS=$((BATCH_SIZE * UPDATE_FREQ))
{
    echo "lr: $LR"
    echo "effective_batch_size: $EFFECTIVE_BS"
    echo "mem_config: $MEM_CONFIG"
    echo "base_ckpt: $CKPT"
} > "$EXP_DIR/experiment_config.yaml"

export PYTHONPATH=$PYTHONPATH:/home/chenty/miniconda3/envs/abacust_mem/bin
echo $PYTHONPATH

run_training() {
      local log_file="${LOG_DIR}/train_memedge16_$(date +%F_%H-%M-%S).log"
      echo "[$(date)] Starting training... log: $log_file"
      CUDA_VISIBLE_DEVICES=0 /home/chenty/miniconda3/envs/abacust/bin/fairseq-train \
            --user-dir /home/chenty/abacust_mem/src/fasde \
            --task diff_full_atom --arch diff_full_atom_base --criterion diff_full_atom_criterion \
            --optimizer adam \
            --lr $LR --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 1000 \
            --max-epoch $MAX_EPOCH \
            --batch-size $BATCH_SIZE \
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
            --config-yaml $CONFIG_YAML \
            --max-protein-sequence-len $MAX_PROTEIN_SEQUENCE_LEN \
            --pdb-path $PDB_PATH \
            --npy-path $NPY_PATH \
            --distributed-backend nccl \
            --pdbtm_file_path $PDBTM_PATH \
            --ca-radius $CA_RADIUS \
            --diff_T $DIFF_T \
            --max_iter_num $MAX_ITER_NUM \
            --find-unused-parameters \
            --embed_unimol_reprs \
            --pretrained_mpnn_ckpt \
            --pretrained_mpnn_ckpt_f $CKPT \
            --ckpt $CKPT \
            --mem_config $MEM_CONFIG \
            --afdb \
            --pdb_ratio 0.25 \
            --afdb_npy_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/npys/all_npy" \
            --afdb_list_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/Nonredundant_merged_cluster_dict.npy" \
            --afdb_tmdet_npy_path "/home/chenty/public_data/abacust_mem_data/tmAFDB_data/tm_npys" \
            --afdb_apply_noise_prob 1 \
            --afdb_noise_sigma 0.2 \
            --tensorboard-logdir $SAVE_DIR/tensorboard \
            --tm_raw $tm_raw \
            --augment_eps 0.2 \
            --nar \
            --esm_pretrained $ESM_PRETRAINED \
            2>&1 | tee -a "$log_file"
}

while true; do
    run_training
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed with code $EXIT_CODE."
        sleep 10
    else
        echo "[$(date)] Training crashed with exit code $EXIT_CODE. Restarting in 20s..."
        sleep 20
    fi
done
