from tqdm import tqdm
from joblib import Parallel, delayed
from utils import download_raw_chain_from_pdbname, read_json

def _download_one(item, output_dir):
    """包装一层，方便 joblib 调用"""
    download_raw_chain_from_pdbname(
        item,
        output_dir=output_dir,
        try_num=3,
        overwrite=False
    )

def main(n_jobs: int = 8):
    input_dir  = "../data_process/data_storage/pdbtm_list.json"
    output_dir = "./fasta"
    pdb_list   = read_json(input_dir)["r_valid"]

    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_download_one)(item, output_dir)
        for item in tqdm(pdb_list, desc="submit")
    )

if __name__ == "__main__":
    main(n_jobs=32)   # 根据机器核心数自己调