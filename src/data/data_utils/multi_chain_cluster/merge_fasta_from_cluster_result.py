from pathlib import Path
from tqdm import tqdm
from utils import read_json   # 你已有的工具函数

def merge_fasta_from_cluster_result(
        input_dir: str,
        query_file: str,
        target_file: str,
        src_file: str):
    """
    把 cluster 结果中 train/valid 对应的 pdb 名找到，
    从 input_dir 读取单个链的 fasta，合并写成 query.fasta 和 target.fasta
    """
    input_dir = Path(input_dir)
    cluster_dict = read_json(src_file)

    # ① 展平拿到 pdb 名列表
    train_pdbs = [pdb for v in cluster_dict["train"].values() for pdb in v]
    valid_pdbs = [pdb for v in cluster_dict["valid"].values() for pdb in v]

    # ② 写入 query
    with open(query_file, "w") as fq:
        for pdb in tqdm(valid_pdbs, desc="build query"):
            fasta_path = input_dir / f"{pdb}.fasta"
            if not fasta_path.exists():
                print(f"[WARN] skip missing {fasta_path}")
                continue
            fq.write(fasta_path.read_text())

    # ③ 写入 target
    with open(target_file, "w") as ft:
        for pdb in tqdm(train_pdbs, desc="build target"):
            fasta_path = input_dir / f"{pdb}.fasta"
            if not fasta_path.exists():
                print(f"[WARN] skip missing {fasta_path}")
                continue
            ft.write(fasta_path.read_text())

    print(f"Done! target={len(train_pdbs)}, query={len(valid_pdbs)}")


if __name__ == "__main__":
    merge_fasta_from_cluster_result(
        input_dir="./fasta",
        query_file="query.fasta",
        target_file="target.fasta",
        # src_file="/home/chenty/abacust_mem/src/data_utils/data_process/data_storage/merged_cluster_dict.json"
        src_file="/home/chenty/public_data/abacust_mem_data/data_storage/merged_cluster_dict.json"
    )