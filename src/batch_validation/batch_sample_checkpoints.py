#!/usr/bin/env python3
"""
批量运行每隔5个检查点的 batch_sample
使用 abacust_mem_zero_with_0124_cluster_dict_b64 的权重
"""

import os
import sys
import subprocess
from multiprocessing import Process
from pathlib import Path

# 添加 abacust_mem 到路径
sys.path.insert(0, '/home/chenty/abacust_mem/src')

# 延迟导入，确保路径已设置
# from batch_validation.batch_sample import abacust_mem_design

# 配置
EXPERIMENT_DIR = "/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64"
CHECKPOINT_DIR = f"{EXPERIMENT_DIR}/checkpoint"
DATASET = "third_cluster_data"

# 每隔约10个检查点选择一个（从实际存在的检查点中）
SELECTED_CHECKPOINTS = [
    10,   # checkpoint10.pt
    20,   # checkpoint20.pt  (间隔10)
    33,   # checkpoint33.pt  (间隔13，最接近30)
    70,   # checkpoint70.pt  (间隔37，中间无40-60的检查点)
    80,   # checkpoint80.pt  (间隔10)
    91,   # checkpoint91.pt  (间隔11，最接近90)
    100,  # checkpoint100.pt (间隔9)
]

# GPU 分配（使用空闲的 GPU 3,4 和其他）
# 可用 GPU: 0,1,2,3,4,5,6 (7个)
GPU_ASSIGNMENTS = [3, 4, 5, 6, 0, 1, 2]  # 优先使用空闲的 3,4

def run_single_checkpoint(checkpoint_num, gpu_id):
    """运行单个检查点"""
    # 延迟导入
    from batch_validation.batch_sample import abacust_mem_design
    
    ckpt_path = f"{CHECKPOINT_DIR}/checkpoint{checkpoint_num}.pt"
    
    # 输出目录 - 使用原始路径结构，按检查点分子目录
    output_base = f"/home/chenty/public_data/{DATASET}/abacust_design_results"
    design_seq_dir = f"{output_base}/seqs_ckpt{checkpoint_num}"
    npy_dir = f"{output_base}/npys"  # 使用共享的npy目录
    
    # 输入数据
    prot_dir = f"/home/chenty/public_data/{DATASET}/source_data/pdbs"
    
    # 从 merged_cluster_dict 获取数据列表
    from protein_utils.pdb_parser import Pdb_processer
    data_names = Pdb_processer.utils.extract_data_list_from_merged_cluster_dict(
        f"/home/chenty/public_data/{DATASET}/source_data/merged_cluster_dict.npy",
        split="valid", mode="all"
    )
    
    print(f"[Checkpoint {checkpoint_num}] Starting on GPU {gpu_id}")
    print(f"  CKPT: {ckpt_path}")
    print(f"  Output: {design_seq_dir}")
    
    try:
        abacust_mem_design(
            input_path=prot_dir,
            design_seq_dir=design_seq_dir,
            npy_dir=npy_dir,
            data_names=data_names,
            tm_raw=f"zero_0124_ckpt{checkpoint_num}",  # tm_raw 标识
            batchsize=10,
            device_list=[gpu_id],
            temperature=0.1,
            iter_num=20,
            ckpt=ckpt_path,
        )
        print(f"[Checkpoint {checkpoint_num}] Completed!")
    except Exception as e:
        print(f"[Checkpoint {checkpoint_num}] Error: {e}")


def main():
    print("=" * 70)
    print("Batch Sample for 0124 Experiment Checkpoints")
    print("=" * 70)
    print(f"Experiment: {EXPERIMENT_DIR}")
    print(f"Selected checkpoints: {SELECTED_CHECKPOINTS}")
    print(f"GPU assignments: {GPU_ASSIGNMENTS[:len(SELECTED_CHECKPOINTS)]}")
    print("=" * 70)
    
    # 验证检查点文件存在
    existing_checkpoints = []
    for num in SELECTED_CHECKPOINTS:
        ckpt_file = f"{CHECKPOINT_DIR}/checkpoint{num}.pt"
        if os.path.exists(ckpt_file):
            existing_checkpoints.append(num)
        else:
            print(f"Warning: {ckpt_file} not found, skipping...")
    
    print(f"\nRunning {len(existing_checkpoints)} checkpoints...")
    
    # 创建进程
    processes = []
    for i, ckpt_num in enumerate(existing_checkpoints):
        gpu_id = GPU_ASSIGNMENTS[i % len(GPU_ASSIGNMENTS)]
        p = Process(target=run_single_checkpoint, args=(ckpt_num, gpu_id))
        processes.append((ckpt_num, p))
        p.start()
    
    # 等待所有进程完成
    print("\n" + "=" * 70)
    for ckpt_num, p in processes:
        p.join()
        print(f"[Checkpoint {ckpt_num}] Process finished")
    
    print("=" * 70)
    print("All checkpoints completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
