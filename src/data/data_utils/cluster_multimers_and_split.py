from protein_utils.fasta_utils import Protein_Sequence
import os
import subprocess
import shutil
from sklearn.model_selection import train_test_split
import warnings
import json
import numpy as np
import pickle
import traceback
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple
import logging
from multiprocessing import Pool
from joblib import Parallel, delayed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from split import Split_dataset
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"{datetime.now():%Y-%m-%d_%H-%M}.log")

# 同时输出到终端和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 终端
        logging.FileHandler(log_file, mode="w", encoding="utf-8")  # 文件
    ],
    force=True
)



def filter_with_resolution_and_hydrophobic_width(
        clusters,
        hydrophobic_thres=(25, 40),
        resolution_cutoff=4,
        n_jobs=64,
        pdb_dir='/home/chenty/abacust_mem/src/data/data_storage/assembled_pdbs'
):
    """
    与原函数完全兼容，只是内部并行加速。
    返回: 过滤后的 defaultdict(list)
    """
    min_thickness, max_thickness = hydrophobic_thres
    pdb_path = Path(pdb_dir)

    def _process(protein: str):
        from protein_utils.pdbtm_data_parser import Pdbtm_parser

        protein_l = protein.lower()
        item = Pdbtm_parser._init_from_pdbname(protein_l)
        if not item:
            return None, 'no_item'

        tm_width = item._membrane_thickness if not item.check_soluble() else 0

        try:
            file_path = next(pdb_path.rglob(f'{protein_l}*.ent'))
        except StopIteration:
            raise FileNotFoundError(f'未找到 {protein}*.ent')

        resolution = float(file_path.stem.split('_')[1]) / 100

        # 过滤逻辑
        if 1e-2 < resolution < resolution_cutoff:
            if min_thickness < tm_width < max_thickness:
                return protein, 'keep'
            else:
                return protein, 'res_ok_thick_fail'
        return None,'filtered'

    tasks = [(repr_, p) for repr_, proteins in clusters.items() for p in proteins]
    raw_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process)(p) for _, p in tqdm(tasks, desc='filtering')
    )

    # --- 收集结果 ---
    result = defaultdict(list)
    invalid_resolution = 0
    for (repr_, p), (protein, flag) in zip(tasks, raw_results):
        if flag == 'keep':
            result[repr_].append(protein)
            # logging.info(protein)
        elif flag == 'res_ok_thick_fail':
            invalid_resolution += 1

    # --- 日志 ---
    logging.info(f"cluster center num: {len(result)}")
    logging.info(f"pdb num: {sum(len(v) for v in result.values())}")
    logging.info(f"invalid resolution: {invalid_resolution}")

    return result





class SequenceClusterer():
    def __init__(self, input_fasta_path, mmseqs_output):
        """
        初始化序列聚类器。

        Args:
            sequence_folder (str): 提取的氨基酸序列将被保存的文件夹路径。
            mmseqs_input (str): MMSeqs2 聚类工具的输入文件夹路径。
            mmseqs_output (str): MMSeqs2 聚类工具的输出文件夹路径。
        """

        self.fasta_path = input_fasta_path
        self.mmseqs_output = mmseqs_output
        os.makedirs(mmseqs_output, exist_ok=True)



    def cluster_sequences(self, cov: float = 0.8, id: float = 0.3) :
        """
        用 MMSeqs2 对 self.fasta_path 中的序列聚类。
        cov: 最小覆盖度，默认 0.8
        id : 最小序列 identity，默认 0.3
        返回: {cluster_id: [members,...]}
        """
        import shutil, subprocess
        from pathlib import Path

        root = Path(self.mmseqs_output)
        logging.info(f"emptying {root}")
        shutil.rmtree(root, ignore_errors=True)

        db_path      = root / "db" / "db"
        clust_path   = root / "cluster"
        tmp_path     = root / "tmp"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        clust_path.mkdir(parents=True, exist_ok=True)
        tmp_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 建库
            logging.info("Creating MMSeqs2 database...")
            subprocess.run(["mmseqs", "createdb", str(self.fasta_path), str(db_path)], check=True)

            # 2. 聚类（coverage + identity）
            logging.info(f"Clustering : cov={cov}, id={id}, cov-mode=0...")
            subprocess.run([
                "mmseqs", "cluster",
                str(db_path), str(clust_path), str(tmp_path),
                "--min-seq-id", str(id),
                "--cov-mode", "0",
                "-c", str(cov)
            ], check=True)

            # 3. TSV
            tsv = clust_path / "clusters.tsv"
            subprocess.run([
                "mmseqs", "createtsv",
                str(db_path), str(db_path), str(clust_path), str(tsv)
            ], check=True)
            return self.read_clusters_tsv(str(tsv))

        except subprocess.CalledProcessError as e:
            logging.warning(f"MMSeqs2 failed: {e}")
            return None

    def read_clusters_tsv(self, clusters_tsv_path, filter_same=True):
        """
        读取并解析 clusters.tsv 文件。

        Args:
            clusters_tsv_path (str): clusters.tsv 文件路径。
            filter_same (bool): 是否过滤重复的序列 ID。默认为 True。

        Returns:
            dict: 聚类结果字典。
        """
        clusters = {}
        try:
            with open(clusters_tsv_path, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            cluster_id, sequence_id = parts[0], parts[1]
                            cluster_id = cluster_id[1:] if cluster_id.startswith(">") else cluster_id
                            sequence_id = sequence_id[1:] if sequence_id.startswith(">") else sequence_id
                            if cluster_id not in clusters:
                                clusters[cluster_id] = []
                            # 如果启用了过滤且序列 ID 已经存在于聚类中，则跳过
                            if filter_same and sequence_id in clusters[cluster_id]:
                                continue
                            clusters[cluster_id].append(sequence_id)
                        else:
                            print(f"Unexpected line format in cluster file: {line.strip()}")
            return clusters
        except FileNotFoundError:
            print(f"File not found: {clusters_tsv_path}")
            return None
        except Exception as e:
            print(f"Error reading cluster file: {e}")
            return None

    def save_clusters_to_json(self, clusters, output_dir=None, filename="clusters.json"):
        """
        将聚类结果字典保存为 JSON 文件。

        Args:
            clusters (dict): 聚类结果字典。
            output_dir (str): 输出目录路径。
            filename (str): 输出文件名。默认为 "clusters.json"。

        Returns:
            str: JSON 文件的完整路径，或 None 表示失败。
        """
        filename = filename if filename.endswith(".json") else filename + ".json"

        if output_dir is None:
            output_dir = self.mmseqs_output
        try:
            os.makedirs(output_dir, exist_ok=True)
            json_file_path = os.path.join(output_dir, filename)
            with open(json_file_path, "w") as json_file:
                json.dump(clusters, json_file, indent=4)
            print(f"Cluster results saved to JSON file: {json_file_path}")
            return json_file_path
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return None

    def save_keys_to_txt(self, clusters, output_dir=None, filename="cluster_center.txt"):
        """
        将聚类结果字典的键保存到文本文件中。

        Args:
            clusters (dict): 聚类结果字典。
            output_dir (str): 输出目录路径。
            filename (str): 输出文件名。默认为 "keys.txt"。

        Returns:
            str: 文本文件的完整路径，或 None 表示失败。
        """

        filename = filename if filename.endswith(".txt") else filename + ".txt"
        if output_dir is None:
            output_dir = self.mmseqs_output
        try:
            os.makedirs(output_dir, exist_ok=True)
            txt_file_path = os.path.join(output_dir, filename)
            with open(txt_file_path, "w") as txt_file:
                for key in clusters.keys():
                    txt_file.write(f"{key}\n")
            print(f"Keys saved to text file: {txt_file_path}")
            return txt_file_path
        except Exception as e:
            print(f"Error saving text file: {e}")
            return None

    def save_clusters_to_npy(self, clusters,
        output_dir='/home/chenty/abacust_mem/src/data/data_storage/',
        filename="merged_cluster_dict" ):
        """保存至npy文件，使用pickle进行读取和书写，存储字典的结构:
            merged_cluster_dict = {
                rep_data_name_1: {
                    'rep': [data_name_1, data_name_2, ..., data_name_n]
                },
                rep_data_name_2: {
                    'rep': [data_name_1, data_name_2, ..., data_name_n]
                },
                ...
            }
        Args:
            clusters (_type_): 输入的字典
            output_dir (str, optional): 存储路径. Defaults to '/home/chenty/abacust_mem/src/data/cluster/'.
            filename (str, optional): 存储的文件名. Defaults to "merged_cluster_dict.npy".
        """
        def save_dict_as_npy_with_pickle(data_dict, file_path):
            """使用 pickle 将字典保存为.npy文件"""
            # 序列化字典
            pickled_data = pickle.dumps(data_dict)
            # 将序列化后的数据保存为.npy文件
            np.save(file_path, np.array(pickled_data, dtype=object))

        filename = filename if filename.endswith(".npy") else filename + ".npy"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
        output_path = os.path.join(output_dir,filename)
        #调整格式：
        data_dict = {key: value for key, value in clusters.items()}
        save_dict_as_npy_with_pickle(data_dict, output_path)

        return output_path


def recluster_pdbs_in_multiple_clusters(cluster_dict, cluster_scale_cutoff = 10) :
    """
    分析每个 PDB 编号的全部链是否都在同一个聚类中心。

    参数
    ----
    cluster_dict: {cluster_id: ['pdb1_A', 'pdb1_B', ...], ...}


    """
    #首先收集统计信息：

    pdb_chains = defaultdict(set)          # pdb -> {repr_set}
    for repr_, items in cluster_dict.items():
        for item in items:
            pdbname = item[:4].lower()
            pdb_chains[pdbname].add(repr_)   # 把所在中心 ID 记入集合

    logging.info(f"total protein number: {len(pdb_chains)}")

    pdb_in_one_cluster   = {p:frozenset(repr_) for p, repr_ in pdb_chains.items() if len(repr_) == 1}
    pdb_in_multi_cluster = {p:reprs for p, reprs in pdb_chains.items() if len(reprs) > 1}

    logging.info(f"========== integrating informations =========")

    logging.info(f"proteins belong to single cluster center : {len(pdb_in_one_cluster)}")
    logging.info(f"proteins belong to multiple cluster center : {len(pdb_in_multi_cluster)}")

    multi_chain_cluster_dict = defaultdict(set)
    for p, reprs in pdb_in_multi_cluster.items():
        multi_chain_cluster_dict[frozenset(reprs)].add(p)

    single_pdb = [v for k,v in multi_chain_cluster_dict.items() if len(v) == 1]

    logging.info(f"cluster_centers with more than one member::{len(multi_chain_cluster_dict.keys())}")
    logging.info(f"cluster_centers with one member :{len(single_pdb)}")
    logging.info("===============================================")


    result = defaultdict(lambda:[])
    for p, repr_ in pdb_in_one_cluster.items():
        result[repr_].append(p)
    for reprs, p in multi_chain_cluster_dict.items():
         if len(reprs) <= cluster_scale_cutoff:
             result[reprs].extend(list(p))

    return result



def run_sequence_clustering(
    base_path='/home/chenty/abacust_mem/src/data/data_storage',                                                      #存放fasta文件和mmseq的地方
    input_fasta_path='/home/chenty/abacust_mem/src/data/data_storage/sequence_for_cluster.fasta',                    #输入文件
    output_dir='/home/chenty/abacust_mem/src/data/data_storage',                                                     #存放最终的npy文件的地方
    output_file_name = "raw_cluster_dict",                                                                        #输出文件名称
    save_json=False,
    save_txt=False
):
    # 初始化聚类器
    sequence_clusterer = SequenceClusterer(
        input_fasta_path=input_fasta_path,
        mmseqs_output=os.path.join(base_path, "mmseqs/output")
    )

    #抽提fasta文件并放到fasta目录下
    clusters = sequence_clusterer.cluster_sequences()

    # 保存结果
    if clusters:
        #########################################################################################################
        total_seqs = sum(len(members) for members in clusters.values())
        num_centers = len(clusters)
        logging.info(f"Total sequences: {total_seqs}")
        logging.info(f"Cluster centers: {num_centers}")
        #######################################################################################################
        if save_json:
            json_path = sequence_clusterer.save_clusters_to_json(clusters, output_dir=output_dir, filename= output_file_name)
            logging.info(f"Clusters saved to {json_path}")

        if save_txt:
            txt_file_path = sequence_clusterer.save_keys_to_txt(
                                    clusters,output_dir=output_dir,filename="cluster_center.txt")
            logging.info(f"Clusters saved to {txt_file_path}")

        npy_path = sequence_clusterer.save_clusters_to_npy(
                                                clusters,
                                                output_dir=output_dir,
                                                filename=output_file_name
                                                )
        logging.info(f"Clusters saved to {npy_path}")
        return clusters
    else:
        logging.warning("Clustering failed.")
        import pdb;pdb.set_trace()

def check_pdbnames_with_leakage(input_path, tmp_dir = "/home/chenty/abacust_mem/src/data/data_storage/tmp", thres = 0.5, cov = 0.8):
    input_path = Path(input_path)
    tmp_dir = Path(tmp_dir)

    clusters = np.load(input_path, allow_pickle = True).item()
    train_data = [value for key, values in clusters["train"].items() for value in values]
    valid_data = [value for key, values in clusters["valid"].items() for value in values]
    train_dir = tmp_dir/"train_dir"
    valid_dir = tmp_dir/"valid_dir"
    # train_dir.mkdir(parents=True, exist_ok=True)
    # valid_dir.mkdir(parents=True,exist_ok=True)
    for d in (train_dir, valid_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=False)
        d.mkdir(parents=True, exist_ok=True)
        logging.warning(f"{d} dir has been emptied ")

    from protein_utils.fasta_utils import Protein_Sequence
    from joblib import Parallel, delayed

    def get_fasta_sequence_wrap(name, output_dir, num):
        source_dir = Path('/home/chenty/abacust_mem/src/data/data_storage/fasta_download')
        name = name.lower()
        source_file = source_dir / f"{name}.fasta"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if source_file.is_file():
            try:
                shutil.copy2(source_file, output_path / f"{name}.fasta")
            except Exception as e:

                logging.error(f"❌ 复制失败 {name}: {e}")
                Protein_Sequence.fasta_utils.download_raw_fasta_file_from_pdbname(name, output_dir, num)
        else:
            logging.info(f"🔄 文件不存在，正在下载 {name}.fasta")
            Protein_Sequence.fasta_utils.download_raw_fasta_file_from_pdbname(name, source_dir, num)
            if source_file.is_file():
                shutil.copy2(source_file, output_path / f"{name}.fasta")


    Parallel(n_jobs=16)(delayed(get_fasta_sequence_wrap)(i, train_dir, 3) for i in tqdm(train_data))
    Parallel(n_jobs=16)(delayed(get_fasta_sequence_wrap)(i, valid_dir, 3) for i in tqdm(valid_data))

    Protein_Sequence.fasta_utils.gather_all_fastas_from_dir(input_dir= train_dir, output_file_name="/home/chenty/abacust_mem/src/data/data_storage/target.fasta", header_rewrite_function = lambda h: h.replace('>', '').split('|')[0])
    Protein_Sequence.fasta_utils.gather_all_fastas_from_dir(input_dir= valid_dir, output_file_name="/home/chenty/abacust_mem/src/data/data_storage/query.fasta", header_rewrite_function = lambda h: h.replace('>', '').split('|')[0])
    from filter_leakage import LeakChecker
    checker = LeakChecker(
        query_fasta="/home/chenty/abacust_mem/src/data/data_storage/query.fasta" ,
        target_fasta="/home/chenty/abacust_mem/src/data/data_storage/target.fasta" ,
        out_dir="/home/chenty/abacust_mem/src/data/data_storage/check_leakage/"
        )
    return checker.run(identity = thres, cov = cov)

def filter_valid_pdbs(split_result):
    filtered_valid_dict = dict()
    train_cluster_centers = {v for vs in split_result["train"].keys() for v in vs.split(",")}
    # logging.info(train_cluster_centers)
    for cluster_center in split_result["valid"]:
        # logging.info(cluster_center)
        if all(element not in train_cluster_centers for element in cluster_center.split(",")):
            filtered_valid_dict.update({cluster_center: split_result["valid"][cluster_center]})

    filtered_result = dict()

    filtered_result["valid"] = filtered_valid_dict
    filtered_result["train"] = split_result["train"]
    return filtered_result



def cluster_multimers(input_fasta_dir,tmp_fasta_name, raw_cluster_result_path, output_dir,split_ratio,filter_identity = 0.5, filter_cov=0.8, cluster_scale_cutoff= 10):
    output_dir = Path(output_dir)
    Protein_Sequence.fasta_utils.gather_all_fastas_from_dir(
        input_dir=input_fasta_dir,  output_file_name=tmp_fasta_name,  length_threshold=[20, 1024],  filter_repeat=20)     #这个阈值是ESM2的最大和最小
    clusters = run_sequence_clustering(
                            input_fasta_path=tmp_fasta_name,
                            output_dir='/home/chenty/abacust_mem/src/data/data_storage',                                                     #存放最终的npy文件的地方
                            output_file_name = "raw_cluster_dict",
                            save_json=True,
                            save_txt=True
                        )
    clusters = recluster_pdbs_in_multiple_clusters(clusters, cluster_scale_cutoff= cluster_scale_cutoff)


    logging.info(f"==============filter with resolution and tm width! ============")
    clusters = filter_with_resolution_and_hydrophobic_width(clusters)
    pdb_number = len([p for center, p_list in clusters.items() for p in p_list])
    logging.info(f"protein number : {pdb_number}")
    logging.info(f"protein cluster center number : {len(clusters)}")

    np.save(raw_cluster_result_path, np.array(dict(clusters), dtype=object))

    logging.info(f"=================  split dataset  ===============")
    split_result, info = Split_dataset.split_dataset_by_clusters(
        file_path=raw_cluster_result_path,  # 假设输入是这里生成的文件
        output_dir=output_dir,
        split_ratio=split_ratio,
        output_format=["json", "npy"],
        seed=42
    )
    logging.info(info)

    logging.info(f"==============diminish valid centers found in train dataset:  ============")
    split_result_with_valid_filtered = filter_valid_pdbs(split_result)
    logging.info(split_result_with_valid_filtered["valid"])
    pdb_number = len([p for center, p_list in split_result_with_valid_filtered["valid"].items() for p in p_list])


    logging.info(f"protein number : {pdb_number}")
    logging.info(f"protein cluster center number : {len(split_result_with_valid_filtered['valid'])}")

    logging.info(f"================  check_leakage ==============")
    np.save(output_dir / "filtered_cluster_dict.npy", split_result_with_valid_filtered)
    report_csv_path = check_pdbnames_with_leakage(output_dir / "filtered_cluster_dict.npy", tmp_dir = "/home/chenty/abacust_mem/src/data/data_storage/tmp", thres = filter_identity, cov=filter_cov)

    from file_utils import read_csv_to_dict, save_dict_as_json

    leakage_dict_ = read_csv_to_dict(report_csv_path)          # {idx: [col0, col1, ...]}
    leakage_pdbs = {row[0] for row in leakage_dict_.values()if len(row) > 0 and row[1] == "leaked"}

    result = {
        "train": split_result_with_valid_filtered["train"],
        "valid": defaultdict(list)
        }

    for center, items in split_result_with_valid_filtered["valid"].items():
        result["valid"][center] = [it for it in items if it not in leakage_pdbs]
    result["valid"] = dict(result["valid"])

    np.save(output_dir / "merged_cluster_dict.npy", result)
    save_dict_as_json(output_dir/"merged_cluster_dict.json", result)

    pass

if __name__ == "__main__":
    cluster_multimers(
        input_fasta_dir = "/home/chenty/abacust_mem/src/data/data_storage/fasta_for_cluster",
        tmp_fasta_name =  "/home/chenty/abacust_mem/src/data/data_storage/fasta_for_cluster.fasta",
        raw_cluster_result_path = "/home/chenty/abacust_mem/src/data/data_storage/processed_raw_cluster_result.npy",
        output_dir = "/home/chenty/abacust_mem/src/data/data_storage",
        split_ratio = 0.85,
        filter_cov=0.8,
        filter_identity=0.5,
        cluster_scale_cutoff=15
    )
