#!/bin/bash
# 运行 0124 实验的每隔5个检查点的 batch_sample

set -e

echo "=========================================="
echo "Batch Sample for 0124 Checkpoints"
echo "Current time: $(date)"
echo "=========================================="

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlfold

echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"

# 设置 PYTHONPATH
export PYTHONPATH="/home/chenty/abacust_mem/src:/home/chenty:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# 运行批处理脚本
cd /home/chenty/abacust_mem/src/batch_validation
python batch_sample_checkpoints.py

echo "=========================================="
echo "Completed!"
echo "End time: $(date)"
echo "=========================================="
