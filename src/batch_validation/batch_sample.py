####################################################################################
# 文件源都已配置好，使用的是生成的npy文件夹，之后直接在里面遍历
# 可以配置单卡推理和多卡推理，多卡的化，会在pdbname的目录下生成带有gpuid编号的fasta文件
# 若要汇总的话可以运行同目录下的batch_gather_result.py，这会在pdbname各自的目录下生成总的fasta文件，并生成csv文件记录所有序列的均值和方差便于作图
# 如果要换验证样本，微调模型的版本，需要关注所使用的dataset路径是否正确，权重是否正确配置，要将输入中的tm_raw改过来，如果要改变inference的线程，还需要改batchsize，gpuid如果验证的版本之前没有，还要重新生成npy
####################################################################################
import numpy as np
import os
import datetime
import subprocess
from multiprocessing import Process
from pathlib import Path

import sys
import logging
from protein_utils.pdb_parser import Pdb_processer
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)



def log_gpu():
    """
    打印当前 PyTorch 使用的 GPU 的逻辑编号、物理设备名称 和 UUID。
    """
    import torch
    import pynvml
    try:
        device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)

        # 初始化 NVIDIA 管理库
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        # 获取物理信息（无需 decode）
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)

        print(f"[log_gpu] ✅ PyTorch 当前逻辑设备: cuda:{device_index}")
        print(f"[log_gpu] ✅ 映射物理 GPU 名称: {name}")
        print(f"[log_gpu] ✅ 映射物理 GPU UUID : {uuid}")

    except Exception as e:
        print(f"[log_gpu] ⚠️ 无法获取 GPU 信息: {e}")

def log_run_params(tm_raw, ckpt, device_list, batchsize, temperature, iter_num, data_names):
    logging.info("=" * 60)
    logging.info(f"tm_raw:      {tm_raw}")
    logging.info(f"ckpt:        {ckpt}")
    logging.info(f"device_list: {device_list}")
    logging.info(f"batchsize:   {batchsize}")
    logging.info(f"temperature: {temperature}")
    logging.info(f"iter_num:    {iter_num}")
    logging.info(f"num_proteins:{len(data_names)}")
    logging.info("=" * 60)


def abacust_mem_design(
        input_path,  #存放pdb文件的dir
        design_seq_dir, #存放结果的路径
        npy_dir,
        data_names,    #一个列表，里面是4字母pdbid
        tm_raw,       # abacust的模式，可选zero, raw,pdbtm,pdbtm_r
        batchsize,
        device_list= [1],
        temperature: float = 0.1,
        iter_num: int = 20,
        augment_eps: float = 0.2,
        ckpt = None,
        designed_pos: str = "",
        mode = "all"
        ):


    ckpt_basename   = os.path.splitext(os.path.basename(ckpt))[0]
    savedir         = design_seq_dir
    Path(savedir).mkdir(parents=True, exist_ok=True)#创建保存路径
    Path(npy_dir).mkdir(parents=True, exist_ok=True)
    log_run_params(tm_raw, ckpt, device_list, batchsize, temperature, iter_num, data_names)
    # ---- 2. 默认值 ----
    name            = f"abacust_mem_{tm_raw}"
    suffix          = f"esm_refined_{tm_raw}/{ckpt_basename}"
    ###################################################################
    # 生成对应的npy文件
    # script_path = "/home/chenty/abacust_mem/src/data/data_utils/make_feature_from_pdb.py"
    #             # "/home/chenty/abacust_mem/src/data_utils/data_process/make_feature_from_pdb.py"
    # command = [
    #     "python", script_path,
    #     "--pdb_dir", input_path,
    #     "--out_dir", npy_dir,
    #     "--protein_chain", "''",
    #     "--atomized_chain", "''",
    #     "--mode", 'train'
    # ]
    # cmd_str = " ".join(command)
    # logging.info(f"command:{cmd_str}")
    # try:
    #     result = subprocess.run(
    #         cmd_str,
    #         shell=True
    #         )
    #     print(" npys has been generated")
    #     # print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print(f"failed to convert files, {e}")
    ###################################################################

    num_gpu = len(device_list)
    # data_list = os.listdir(f"{npy_dir}/all_npy")
    data_list = [f"{data_name}.npy" for data_name in data_names]
    data_chunks = [data_list[i::num_gpu] for i in range(num_gpu)]
    # print(data_chunks)

    assert len(data_chunks) == num_gpu, f" number of gpus is inconsistent with num of data_chunk"


    processes = []
    for i, data_item in enumerate(data_chunks, start=0):
        order_suffix = i
        device = device_list[i]  # 对 GPU 数量取余，确保循环使用 GPU
        # 启动子进程来执行任务
        p = Process(target=run_design_on_device, args=(data_item,  #单个gpu要处理的数据的序列
                                                       order_suffix,
                                                       device,
                                                       iter_num, temperature, suffix,
                                                       npy_dir,
                                                       ckpt,
                                                       savedir, input_path,
                                                       batchsize,
                                                       designed_pos, augment_eps, tm_raw)
                    )
        processes.append(p)
        p.start()

    # 等待所有子进程完成
    for p in processes:
        p.join()


def run_design_on_device(data_item, order_suffix, device, iter_num, temperature, suffix, npy_dir, ckpt, savedir, input_path, batchsize, designed_pos, augment_eps, tm_raw):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    env['PYTHONPATH'] = '/home/chenty/public_data/tmdet_data/data_utils:/home/chenty/public_utils:' + env.get('PYTHONPATH', '')  # 将数据任务分配给不同的GPU
    gpu_str = str(device)
    logging.info(f"Using device {gpu_str} for data group {order_suffix}")

    cmd = [
        "/home/chenty/miniconda3/envs/abacust/bin/python3", "batch_seq_design_with_lig_raw_pdb.py",  # 确保该脚本在 $PWD 或已在 PYTHONPATH
        "--data_list",":".join(data_item),
        "--order_suffix", str(order_suffix),
        "--iter_num", str(iter_num),
        "--temperature", str(temperature),
        "--suffix", suffix,
        "--npy_dir", os.path.join(npy_dir, "all_npy"),
        "--checkpoint", ckpt,
        "--save_dir", savedir,
        "--consider_lig",
        "--embed_unimol_reprs",
        "--nar",
        "--esm_refinement",
        "--root_dir", input_path,
        "--batchsize", str(batchsize),
        "--designed_positions", designed_pos,
        "--PLM_selfcond", "1",
        "--PLM_param_dir", "/database/lyf_database/pretrain_lm/esm/param",
        "--selfcondPLM", "650M",
        "--augment_eps", str(augment_eps),
        "--mask_mode", "aatype_nll",
        "--write_allatom_model",
        "--tm_raw", tm_raw,
        "--device", f"cuda:0",  # 使用特定的GPU
        "--max_lig_num", "10",
    ]

    subprocess.run(cmd, env=env)




def read_lines_as_list(txt_path):
    """
    读取文本文件的每一行，strip 后作为 list 元素返回

    :param txt_path: 文本文件路径（.txt）
    :return: list[str]，每行为一个元素（去除首尾空白符）
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():

    # 从指定的数据集中读取文件的时候用这个
    dataset = 'third_cluster_data'
    # data_names = read_lines_as_list(f"/home/chenty/public_data/{dataset}/data_list.txt")

    data_names = Pdb_processer.utils.extract_data_list_from_merged_cluster_dict(
        f"/home/chenty/public_data/{dataset}/source_data/merged_cluster_dict.npy",
        split="valid", mode="all"
    )

    output_path = f"/home/chenty/public_data/{dataset}/abacust_design_results/seqs"
    npy_dir = f"/home/chenty/public_data/{dataset}/abacust_design_results/npys"
    prot_dir = f"/home/chenty/public_data/{dataset}/source_data/pdbs"

    runs = [
        {"checkpoint_path": "/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_afdb_with_0124afdb_cluster_dict_1_3_b48/checkpoint/checkpoint260.pt", "device_list": [0], "tm_raw": "zero_afdb_with_0124afdb_cluster_dict_1_3_b48"},
        {"checkpoint_path": "/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_afdb_with_0124afdb_cluster_dict_1_3_b48/checkpoint/checkpoint250.pt", "device_list": [5], "tm_raw": "zero_afdb_with_0124afdb_cluster_dict_1_3_b48"},
        {"checkpoint_path": "/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_afdb_with_0124afdb_cluster_dict_1_3_b48/checkpoint/checkpoint255.pt", "device_list": [6], "tm_raw": "zero_afdb_with_0124afdb_cluster_dict_1_3_b48"},
    ]

    processes = []
    for run in runs:
        checkpoint_path = run["checkpoint_path"]
        device_list = run["device_list"]
        tm_raw = run["tm_raw"]
        p = Process(
            target=abacust_mem_design,
            kwargs=dict(
                input_path=prot_dir,
                design_seq_dir=output_path,
                npy_dir=npy_dir,
                data_names=data_names,
                tm_raw= tm_raw,
                batchsize=10,
                device_list=device_list,
                ckpt=checkpoint_path,
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()




if __name__ == "__main__":
    main()
