#!/usr/bin/env python3
分析DDPM迭代解码中每轮固定氨基酸的特征
import os
import sys
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

# 添加路径
sys.path.insert(0, '/home/chenty/abacust_mem/src/fasde')
sys.path.insert(0, '/home/chenty/miniconda3/envs/abacust/lib/python3.8/site-packages')

# 氨基酸映射
raw_restypes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
restype_order = {r: i+4 for i, r in enumerate(raw_restypes)}  # 4-23是标准氨基酸
res_id_to_aatype = {v: k for k, v in restype_order.items()}

def analyze_fixed_positions(iter_seq_list, initial_mask):
    
