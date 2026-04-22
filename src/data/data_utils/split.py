#划分训练集与测试集，根据聚类结果对每一个类按照3：7划分，如果只有一个样本的话，则统一化归训练集

import json
import numpy as np
import os
import math

from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import numpy as np
import pickle

def load_npy_dict(file_path):

    # item():array(dict)-->dict
    return np.load(file_path, allow_pickle=True).item()

class Split_dataset:
    def __init__(self):
        logging.info(f"class Split_dataset has been initialized...")
        pass
    @classmethod
    def split_dataset_by_clusters( cls, file_path = '', output_dir = '', split_ratio = 0.7, output_format = ["json", "npy"], seed = 42, output_file_name = "merged_cluster_dict"):
        """
        按聚类划分数据集，将 split_ratio 的聚类分入训练集，剩余划入验证集。

        Args:
            file_path (str): 输入 JSON 文件路径，格式为 {cluster_id: [samples]}。
            output_file_path (str): 输出存放的目录
            split_ratio (float): 训练集聚类比例，默认 0.7。

        Returns:
            str: 保存的文件路径
        """
        assert file_path.endswith('.npy'), "Input must be a npy file"
        # import pdb;pdb.set_trace()
        cluster_data = load_npy_dict(file_path)
        cluster_data = {','.join(sorted(k)) if isinstance(k, frozenset) else k: v
          for k, v in cluster_data.items()}
        cluster_ids = list(cluster_data.keys())
        np.random.seed(seed);
        np.random.shuffle(cluster_ids)
        split_index = int(len(cluster_ids) * split_ratio)
        result = {
            "train": {cid: cluster_data[cid] for cid in cluster_ids[:split_index]},
            "valid": {cid: cluster_data[cid] for cid in cluster_ids[split_index:]}
        }

        if "json" in output_format:
            output_file_path = os.path.join(output_dir, f'{output_file_name}.json')
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file_path, 'w') as f:
                json.dump(result, f, indent=4)

            logging.info(f"split result has been saved into {output_file_path}")

        if 'npy' in output_format:
            output_file_path = os.path.join(output_dir, f'{output_file_name}.npy')
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            np.save(output_file_path, result)

            logging.info(f"split result has been saved into {output_file_path}")

        elif "json" not in output_format and "npy" not in output_format:
            logging.warning(f"output format mast in json or npy but got {output_format}, no files will be saved")

        info = {
            "train_cluster_num" : len(cluster_ids[:split_index]),
            "valid_cluster_num" : len(cluster_ids[split_index:]),
            "train_data_num" : sum(len(item) for item in result["train"].values()),
            "valid_data_num" : sum(len(item) for item in result["valid"].values())
        }
        return result, info




# 示例用法
if __name__ == "__main__":
    json_file = "/home/chenty/abacust_mem/src/data_preprocessor/mmseqs/output/clusters.json"  # 输入 JSON 文件路径
    output_dir = "/home/chenty/abacust_mem/src/data/split"  # 输出目录
    split_dataset = Split_dataset()
    split_dataset.split_dataset_by_clusters(json_file, output_dir)
