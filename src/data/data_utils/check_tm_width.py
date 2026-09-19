####################################################################################
# 2025.11.23记
# 利用pdbtmparser里面写好的批量分析函数，生成csv文件，分析膜厚度的总体特征

######################################################################################
import os
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd  # 用于保存 CSV 和 Excel
import json
from protein_utils.pdbtm_data_parser import Pdbtm_parser

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)





# 示例用法
if __name__ == '__main__':
    input_dir =    '/home/chenty/public_data/abacust_mem_data/data_storage/tmdet_result/jsons' # 替换为实际文件路径
    save_csv_dir = '/home/chenty/public_data/abacust_mem_data/data_storage' # 保存 CSV 的目录

    Pdbtm_parser.batch_analyse_tm_width(input_dir=input_dir, save_csv_dir=save_csv_dir, n=32)