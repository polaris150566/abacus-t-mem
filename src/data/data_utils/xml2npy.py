# TmDet region code → internal integer (for ML/feature encoding)
REGION_CODE_MAP = {
    "H": 1,  # Transmembrane helix
    "B": 2,  # Transmembrane beta barrel
    "L": 3,  # Re-entrant loop
    "F": 4,  # Interfacial helix
    "N": 5,  # Beta-barrel inside
    "1": 6,  # Side-one (extracellular/cytosolic)
    "2": 7,  # Side-two (opposite side)
    "3": 8,  # Periplasm / inter-membrane space (double-membrane)
    "P": 9,  # False-positive membrane (fragment analysis)
    "R": 10, # False-negative membrane (fragment analysis)
}
import json
import os
from pathlib import Path
import numpy as np
import xmltodict
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import re



def strip_at_and_copyright(obj,mode = "json"):
    """递归去掉 key 前的 '@'，并删除 copyright 分支"""
    # fn = obj['pdbtm']['@pdbCode']
    # pdbname = re.match(r'([0-9a-zA-Z]{4})\.ent$', fn)
    assert mode in ["json", "npy"], f"mode must in json and npy but got {mode}"
    if isinstance(obj, list):
        # import pdb;pdb.set_trace()
        return [strip_at_and_copyright(item,mode=mode) for item in obj]
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            try:
                new_k = k[1:] if k.startswith('@') else k

                #  个性化处理各种键值的命名规范

                if new_k.lower() == 'copyright':continue
                elif new_k.startswith("xmlns"):   continue
                elif new_k.startswith("xsi:"):    continue
                elif new_k == "translate":
                    matrix = np.array([float(v[f"@{k}"]) for k in 'xyz'])
                    if mode == "json":
                        matrix = matrix.tolist()
                    v = matrix
                elif new_k == "rotate":
                    matrix = np.array([[float(v[f][f"@{k}"]) for k in 'xyz'] for f in ('rowX', 'rowY', 'rowZ')],  dtype=np.float32)
                    assert abs(np.linalg.det(matrix) - 1.0) < 1e-3, f'det = {np.linalg.det(matrix):.6f} but not 1! '
                    if mode == "json":
                        matrix = matrix.tolist()
                    v = matrix

                elif new_k.lower() == "sequence":
                    if v:
                        v = ''.join(v.split(None))
                    else:
                        # print(f"{pdbname}:sequence is empty")
                        print(k,v)
                if new_k == "chain":
                    v = v if isinstance(v, list) else [v]
                if new_k == "region":
                    v = v if isinstance(v, list) else [v]
                # if new_k.startswith("chain"):
                #     import pdb;pdb.set_trace()
                clean[new_k] = strip_at_and_copyright(v, mode = mode)

            except Exception as e:
                import traceback;traceback.print_exc(e)
                # import pdb;pdb.set_trace()
        return clean
    return obj



def xml_to_json(xml_path: Path, json_dir: Path, indent = 2):
    """
    把单个 XML → JSON（字典）并保存；返回 dict
    """
    with xml_path.open('rb') as f:
        data = xmltodict.parse(f, process_namespaces=False)
        data = strip_at_and_copyright(data,mode = "json")

    out_file = json_dir / f'{xml_path.stem}.json'
    with out_file.open('w', encoding='utf-8') as jf:
        json.dump(data, jf, ensure_ascii=False, indent=indent)
    return data

def xml_to_npy(xml_path: Path, npy_dir: Path):
    """
    把单个 XML → JSON 字典并保存为 .npy 文件；返回 dict
    完全保留你现有的 strip_at_and_copyright 逻辑
    """
    # 这是你原来的处理流程，一行不改
    with xml_path.open('rb') as f:
        data = xmltodict.parse(f, process_namespaces=False)
        data = strip_at_and_copyright(data, mode = "npy")
        # import pdb;pdb.set_trace()

    # 只改这里：保存为 .npy 而不是 .json
    npy_file = npy_dir / f'{xml_path.stem}.npy'
    # 将字典转换为 NumPy 数组，dtype='object' 让 NumPy 支持保存字典
    data_array = np.array([data], dtype=object)

    # 使用 np.save 保存
    np.save(npy_file, data_array)

    return data

def batch_convert(xml_root, json_root, npy_dir, n_jobs = 128):
    """
    批量 xml → json,npy
    """
    json_root.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    xml_files = [p for p in xml_root.iterdir() if p.suffix.lower() == '.xml']
    results = Parallel(n_jobs=n_jobs, backend='threading')( delayed(xml_to_npy)(f, npy_dir) for f in tqdm(xml_files, desc='XML→NPY'))
    results = Parallel(n_jobs=n_jobs, backend='threading')( delayed(xml_to_json)(f, json_root) for f in tqdm(xml_files, desc='XML→JSON'))
    return results

# if __name__ == '__main__':
#     input_path = Path('/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/xml')
#     json_path  = Path('/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/jsons')
#     npy_path   = Path('/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/npys')  # 后续 tensor 用

#     batch_convert(input_path, json_path,npy_path, n_jobs=32)

import argparse
from pathlib import Path

def main(input_path, json_path, npy_path, n_jobs):
    batch_convert(input_path, json_path, npy_path, n_jobs=n_jobs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量转换文件')
    parser.add_argument('--input', type=Path, required=True, help='输入xml路径')
    parser.add_argument('--json', type=Path, required=True, help='输出json路径')
    parser.add_argument('--npy', type=Path, required=True, help='输出npy路径')
    parser.add_argument('--jobs', type=int, default=32, help='并行任务数(默认32)')

    args = parser.parse_args()

    main(args.input, args.json, args.npy, args.jobs)