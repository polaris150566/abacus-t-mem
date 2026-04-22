#!/bin/bash
# ------------- 通用变量 -------------
DATA_DIR="/home/chenty/abacust_mem/src/fasde/data"
# tm_raw='raw_esm2_uniprotkb_pdbtm_merged_lora650M_tmdet1'
tm_raw="zero_afdb_with_0223_cluster_dict_1_3_b64"
# 决定权重的加载和保存
# CKPT='/home/chenty/abacust_mem/src/experiments/abacust/checkpoint/checkpoint_best_mem_raw650M.pt'
CKPT='/home/chenty/abacust_mem/src/experiments/abacust_mem_zero/checkpoints/checkpoint_best_noise650M.pt'
# CKPT=/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}/checkpoint/checkpoint_best_noise650M.pt
SAVE_DIR=/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}/checkpoints
# SAVE_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_pdbtm/checkpoints"
# SAVE_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_pdbtm_r/checkpoints"
LOG_DIR="/home/chenty/abacust_mem/src/scripts/log"
LOG_FILE="${LOG_DIR}/train_$(date +%F_%H-%M-%S).log"

mkdir -p $LOG_DIR
mkdir -p $SAVE_DIR
chmod 755 $SAVE_DIR

# ------------- 训练超参 -------------
LR=5e-5
MAX_EPOCH=2000
BATCH_SIZE=4
NUM_WORKERS=4 #4，打算用多少个子进程来并行加载数据
UPDATE_FREQ=16 # 梯度积累
LOG_INTERVAL=100
SAVE_INTERVAL=1
VALIDATE_INTERVAL=1
KEEP_LAST_EPOCHS=20
PATIENCE=5
SEED=42

CONFIG_YAML="config.yaml"
MAX_PROTEIN_SEQUENCE_LEN=256
PDB_PATH="/home/chenty/abacust_mem/src/data/data_storage/assembled_pdbs/"
PDBTM_PATH='/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/npys/'
NPY_PATH='/home/chenty/abacust_mem/src/data/data_storage/merged_npys/all_npy'

DIFF_T=40
MAX_ITER_NUM=4
CA_RADIUS=9.0 # 定义的 ca 原子半径， c 的原子半径 91pm，共价半径 77pm，范德华半径 170pm

ESM_PRETRAINED="/home/chenty/abacust_mem/src/experiments/esm/esm2_t33_650M_UR50D.pt"

# ------------- 保存实验配置到实验目录 -------------
EXP_DIR="/home/chenty/abacust_mem/src/experiments/abacust_mem_${tm_raw}"
EFFECTIVE_BS=$((BATCH_SIZE * UPDATE_FREQ))
{
    echo "lr: $LR"
    echo "effective_batch_size: $EFFECTIVE_BS"
} > "$EXP_DIR/experiment_config.yaml"

# ------------- Python 路径 -------------
export PYTHONPATH=$PYTHONPATH:/home/chenty/miniconda3/envs/abacust_mem/bin
echo $PYTHONPATH

# ------------- screen 会话名 -------------
SESSION_NAME="fairseq_train_abacust_mem_"

# ------------- 启动函数 -------------
run_training() {
      local log_file="${LOG_DIR}/train_$(date +%F_%H-%M-%S).log"
      echo "[$(date)] Starting training... log: $log_file"
      # 设置CUDA_VISIBLE_DEVICES为多个GPU（例如0,1,2,3）
      CUDA_VISIBLE_DEVICES=3 /home/chenty/miniconda3/envs/abacust/bin/fairseq-train \
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
            --afdb \
            --pdb_ratio 0.25 \
            --afdb_npy_path "/home/chenty/abacust_mem/src/data/tmAFDB_data/npys/all_npy" \
            --afdb_list_path "/home/chenty/abacust_mem/src/data/tmAFDB_data/Nonredundant_merged_cluster_dict.npy" \
            --afdb_tmdet_npy_path "/home/chenty/abacust_mem/src/data/tmAFDB_data/tm_npys" \
            --afdb_apply_noise_prob 1 \
            --afdb_noise_sigma 0.2 \
            --tensorboard-logdir $SAVE_DIR/tensorboard \
            --tm_raw $tm_raw \
            --augment_eps 0.2 \
            --nar \
            --esm_pretrained $ESM_PRETRAINED \
            2>&1 | tee -a "$log_file"
}

# ------------- 守护循环 -------------
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


# 后台运行：screen -S fairseq_train_abacust_mem_
# conda activate torch101_0
# /home/chenty/abacust_mem/src/scripts/train.sh

# 进入后台：                screen -r fairseq_train_abacust_mem_
# tensorboard 查看进度：    tensorboard --logdir=/home/chenty/abacust_mem/src/experiments/abacust_mem_raw/checkpoints/tensorboard
#                          tensorboard --logdir=/home/chenty/abacust_mem/src/experiments/abacust_mem_pdbtm/checkpoints/tensorboard
#                          tensorboard --logdir=/home/chenty/abacust_mem/src/experiments/abacust_mem_pdbtm_r/checkpoints/tensorboard --port=6011
# 查看日志：                tail -f ${SAVE_DIR}/train_*.log
