#!/bin/bash

# 设置参数
# PDB_DIR="/home/chenty/abacust_mem/src/data/data_storage/assembled_pdbs"  # 设置 PDB 文件目录
# OUT_DIR="/home/chenty/abacust_mem/src/data/data_storage/merged_npys"  # 设置输出目录

PDB_DIR="/home/chenty/abacust_mem/src/data/tmAFDB_data/pdbs"  # 设置 PDB 文件目录
OUT_DIR="/home/chenty/abacust_mem/src/data/tmAFDB_data/npys"  # 设置输出目录

PROTEIN_CHAIN=""  # 设置蛋白质链，默认为全链
ATOMIZED_CHAIN=""  # 设置原子化链，默认为空
MERGE_ATOMIZED_CHAIN=False  # 是否合并原子化链，默认为 false

mkdir -p $OUT_DIR

# 运行 Python 脚本
python ./make_feature_from_pdb.py \
  --pdb_dir "$PDB_DIR" \
  --out_dir "$OUT_DIR" \
  --protein_chain "$PROTEIN_CHAIN" \
  --atomized_chain "$ATOMIZED_CHAIN" \
  --mode "train"