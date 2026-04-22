import glob
import re

from multiprocessing import Pool
import numpy as np
import csv
import os
import sys
import MDAnalysis as mda
from MDAnalysis.analysis import align
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def batch_calculate_rmsd(select = False,position = None):
    #遍历当前目录中af2_results文件夹下的所有文件夹（里面是af2的预测结果）之后取出每个文件夹下的relaxed_model_1.pdb文件，
    # 这几个文件夹中有一个叫做template，把这个文件夹中的relaxed_model_1.pdb和其他几个文件夹中的同名文件进行比对
    # 获取当前工作目录下的 af2_results 目录
    base_dir = os.getcwd()  # 当前目录
    af2_results_dir = os.path.join(base_dir, 'af2_results')


    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(af2_results_dir) if os.path.isdir(os.path.join(af2_results_dir, f))]

    # 找到 template 文件夹并提取其中的 relaxed_model_1.pdb
    template_folder = None
    template_structure = None

    for folder in subfolders:
        if 'template' in folder:
            template_folder = folder
            template_structure = os.path.join(af2_results_dir, template_folder, 'relaxed_model_1.pdb')
            break

    if not template_structure:
        logging.error("未找到 'template' 文件夹或其下的 relaxed_model_1.pdb 文件。")
        return

    #####################################################################################
    if select:
        if position is None:
            logging.info("position mask is none!")
        select_mask1 = position > 1
        select_mask2 = (position >= -1) & (position <= 1)
        select_mask3 = position < -1
        select_masks = [select_mask1, select_mask2, select_mask3]

    ######################################################################################


    # 遍历其他文件夹，计算 RMSD
    rmsd_values = {}
    for folder in subfolders:
        print(folder)
        if 'template' not in folder:

            target_structure = os.path.join(af2_results_dir, folder, 'relaxed_model_1.pdb')
            print(target_structure)
            if os.path.exists(target_structure):

                for i, select_mask in enumerate(select_masks):
                    rmsd = calculate_rmsd(template_structure, target_structure, select = select, select_mask = select_mask)
                    rmsd_values[folder][i] = rmsd

    # 输出 RMSD 结果
    print (rmsd_values)
    for folder, rmsd in rmsd_values.items():
        logging.info(f"与 {folder} 文件夹的 RMSD: {rmsd:.3f}")


def calculate_rmsd(pdb1, pdb2,select = False, select_mask = None):
    """计算两个PDB文件主链原子之间的RMSD值"""
    # 加载两个PDB文件
    select = True
    u1 = mda.Universe(pdb1)
    # import pdb;pdb.set_trace()
    u2 = mda.Universe(pdb2)

    # 选择主链原子（N, CA, C）
    # 确保两个结构有相同数量的原子
    prot1 = u1.select_atoms("name N CA C")
    prot2 = u2.select_atoms("name N CA C")
    if select:
        mask = np.repeat(np.asarray(select_mask, dtype=bool), 3)
        if len(mask) != len(prot1):
            raise ValueError(f"select_mask 长度与主链原子数不符,select_mask是{len(select_mask)}，而prot是{len(prot1)}")


    if len(prot1) != len(prot2):
        raise ValueError(f"PDB files {pdb1} and {pdb2} have different number of backbone atoms")

    # 首先对齐结构
    align.alignto(u1, u2, select="name N CA C", weights="mass")

    # 计算RMSD
    # 使用MDAnalysis的rmsd函数
    if select:
        mobile = prot1.positions[mask]
        target = prot2.positions[mask]
    else:
        mobile = prot1.positions
        target = prot2.positions
    mobile_com = mobile - mobile.mean(axis=0)
    target_com = target - target.mean(axis=0)
    rmsd = np.sqrt(np.mean((mobile_com - target_com)**2))

    return rmsd

def process_files(file1, file2, output_csv):
    """处理两个输入文件并生成输出CSV"""
    # 读取两个输入文件中的PDB路径
    with open(file1, 'r') as f:
        pdb_paths1 = [line.strip() for line in f if line.strip()]

    with open(file2, 'r') as f:
        pdb_paths2 = [line.strip() for line in f if line.strip()]

    # 检查行数是否相同
    if len(pdb_paths1) != len(pdb_paths2):
        raise ValueError("The two input files have different number of lines")

    # 准备输出数据
    results = []
    for i, (pdb1, pdb2) in enumerate(zip(pdb_paths1, pdb_paths2)):
        try:
            rmsd = calculate_rmsd(pdb1, pdb2)
            results.append([os.path.basename(pdb1), os.path.basename(pdb2), rmsd])
            print(f"Processed pair {i+1}: {pdb1} vs {pdb2} - RMSD: {rmsd:.3f}")
        except Exception as e:
            print(f"Error processing pair {i+1}: {pdb1} vs {pdb2} - {str(e)}")
            results.append([os.path.basename(pdb1), os.path.basename(pdb2), "Error"])

    # 写入CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PDB1", "PDB2", "RMSD"])
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

def nodewrapAF2(YOUR_FASTA_FORMAT_SEQ_FILE, out_root, GPU_idx):
    os.system(f'bash /home/chenty/af2_prediction/pred_from_abacustmem/run_af2_local.sh -i {YOUR_FASTA_FORMAT_SEQ_FILE} -g {GPU_idx} -o {out_root}')

def run_prediction(YOUR_FASTA_FORMAT_SEQ_FILE, af2_pred_out_dir, GPU_idx):
    """
    进行蛋白质结构预测，输入直接是文件路径和GPUidx
    """
    try:
        # 这里运行 nodewrapAF2 预测任务
        nodewrapAF2(YOUR_FASTA_FORMAT_SEQ_FILE, af2_pred_out_dir, GPU_idx)
        logging.info(f"Prediction for {YOUR_FASTA_FORMAT_SEQ_FILE} completed successfully.")
    except Exception as e:
        logging.error(f"Error during prediction for {YOUR_FASTA_FORMAT_SEQ_FILE}: {e}")

def run_AFprediction(base_path, output_dir, GPU_idx=1, workers=10):
    """
    base_path : 包含 *.fasta 的目录（会被设为当前工作目录）
    output_dir: 结果输出目录（可绝对/相对）
    GPU_idx   : 供 run_prediction 内部使用
    workers   : 并行进程数
    """
    # 1. 切到目标目录，保证后面 glob 搜的是 base_path 下的文件
    base_path = os.path.abspath(base_path)
    output_dir = os.path.abspath(output_dir)   # ← 在切目录前完成
    original_cwd = os.getcwd()

    # 1. 再切到 base_path
    os.chdir(base_path)

    # 2. 搜集 fasta
    fasta_files = glob.glob("*.fasta")
    if not fasta_files:
        logging.warning("No *.fasta found in %s", base_path)
        os.chdir(original_cwd)
        return

    # 3. 确保输出目录存在（用绝对路径省心）
    os.makedirs(output_dir, exist_ok=True)

    # 4. 进程池并行
    with Pool(processes=workers) as pool:
        for file in fasta_files:                    # file 只是文件名
            logging.info("Processing %s", file)
            pool.apply_async(run_prediction,
                            args=(os.path.abspath(file),   # 把绝对路径传给子进程
                                output_dir,
                                GPU_idx))
        pool.close()
        pool.join()

    # 5. 恢复原来的工作目录
    os.chdir(original_cwd)

def clean_fasta(path = None):
    """
    递归删除指定目录（或当前目录）及其所有子目录下的 .fasta / .fa 文件（大小写不敏感）。
    path : str, optional
        要清理的目录；若省略则使用当前脚本所在目录。
    """
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    # 切换到目标目录
    os.chdir(path)
    for pat in ('*.fasta', '*.fa', '*.FASTA', '*.FA'):
        for fasta_file in glob.glob(pat):
            try:
                os.remove(fasta_file)
                print(f"已删除: {fasta_file}")
            except Exception as e:
                print(f"删除 {fasta_file} 时出错: {e}")

def split_by_chain(base_path,input_file, name, batchsize = 5):
    # clean_fasta()
    print(input_file)
    with open(input_file, 'r') as infile:
        # 读取所有行
        lines = infile.readlines()
    header = ''
    _len = len(lines)
    print(_len)
    for idx, line in enumerate(lines):
        # import pdb;pdb.set_trace()
        line = line.strip()  # 去除首尾空格

        if idx == 0:
            header = name + f"_template"
            continue
        if idx == 1:
            output_file = os.path.join(base_path,f"{header}.fasta")
            # output_file = f"{header}.fasta"
            output_content = []

            chains = re.split(r'[:\|]+', line)
            # 遍历每条链并保存到不同的文件
            for i, chain in enumerate(chains, start=1):
                chain_header = f">{header}_chain_{chr(64 + i)}"  # 生成链的header，A=65，B=66，...
                output_content.append(f"{chain_header}\n{chain}\n") # 生成文件名，如protein1_A.fasta

            with open(output_file, 'w') as outfile:
                output = "".join(output_content)
                outfile.write(output)
                logging.info(f"{output_file} loaded")
        if (_len/2 - batchsize) > idx/2:
            continue
        # 如果是header（以>开头），则继续处理
        if line.startswith(">"):
            i = idx/2 - _len/2 + batchsize +1
            header = name + f"_{i}"  # 获取header的名字，去掉" > "
        else:
            # 处理蛋白质序列，这里假设每个序列的多个链通过冒号分隔
            output_file = os.path.join(base_path,f"{header}.fasta")
            output_content = []
            chains = re.split(r'[:|]+', line) # 根据冒号分隔多条链
            chains = [s for s in chains if s != ""]

            # 遍历每条链并保存到不同的文件
            for i, chain in enumerate(chains, start=1):
                chain_header = f">{header}_chain_{chr(64 + i)}"  # 生成链的header，A=65，B=66，...
                output_content.append(f"{chain_header}\n{chain}\n") # 生成文件名，如protein1_A.fasta

            with open(output_file, 'w') as outfile:
                output = "".join(output_content)
                outfile.write(output)
                print(f"{output_file} loaded")


def main(data_args):
    base_path = os.path.join(data_args["root_dir"], data_args["time"], data_args['suffix'],data_args['data'])#目录位置
    fasta_file = os.path.join(base_path,f"{data_args['data']}_design.fa")
    split_by_chain(base_path,fasta_file, data_args['data'], data_args["batchsize"])

    run_AFprediction(
        base_path,
        os.path.join(data_args['root_dir'],data_args['design_structure_path']),
        GPU_idx = data_args['device'],
        workers = 10
        )
    # select = False
    # position = np.load(os.path.join(position_dir,"position.npy"))
    # batch_calculate_rmsd(select = select, position = position[0])
    # batch_calculate_rmsd(select = select, position = None)
    pass



def get_valid_target_list(blank = False):
    data = np.load('merged_cluster_dict.npy', allow_pickle=True)
    data_dict = data.item()
    data_dict_train = data_dict['train']
    data_dict_valid = data_dict['valid']
    if blank == True:
        data_dict_valid = {k: v for k, v in data_dict_valid.items() if k not in data_dict_train}

    data_list_valid = [v[0] for k ,v in data_dict_valid.items()]
    # print(data_list_valid)
    return data_list_valid #一个列表



if __name__ == '__main__':
    data_list = get_valid_target_list(blank=True)
    batchsize = 1
    device = 0
    DEFAULT_ROOT = "/home/chenty/abacust_mem/src/inference/valid/demo"
    design_structure_path = 'design_structure'
    design_rmsd_path= 'design_rmsd'
    design_seq_path = "design_seq"
    suffix = 'T_0.1_R_20_esm_refined_pdbtm'
    time = "2025-09-08_19:40:28"

    for data in data_list:
        data_args = {
            "data": data,
            "root_dir":DEFAULT_ROOT,
            "time":time,
            "suffix":suffix,
            "design_structure_path":design_structure_path,
            "design_rmsd_path":design_rmsd_path,
            "design_seq_path":design_seq_path,
            "batchsize":batchsize,
            "device":device

        }
        main(data_args)
