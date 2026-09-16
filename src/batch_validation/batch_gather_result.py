#!/usr/bin/env python3
import os
import re, sys
from pathlib import Path
from typing import List, Tuple
import logging
import statistics as st
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
def read_template(path: Path) :
    """只取 fasta 文件第一条序列，返回 [(header, seq)]"""
    headers, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                headers.append(line)
            else:
                seqs.append(line)
                break
    return list(zip(headers, seqs))

def read_last_n(path: Path, n: int) :
    """取 fasta 文件最后 n 条序列，返回 [(header, seq)]"""
    headers, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                headers.append(line)
                seqs.append('')
            else:
                if seqs:
                    seqs[-1] += line
    return list(zip(headers[-n:], seqs[-n:]))

def read_first_n(path: Path, n: int) :
    """取 fasta 文件前 n 条序列，返回 [(header, seq)]"""
    headers, seqs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                headers.append(line)
                seqs.append('')
            else:
                if seqs:
                    seqs[-1] += line
    # 只要最后 n 条
    return list(zip(headers[1:n+1], seqs[1:n+1]))

def gather_simple_results(batch_size, iter_number = 20, root_dir = ''): #root_dir精确到pdbname
    root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
    code = root_dir.name.lower()
    if not re.fullmatch(r'\w{4}', code):
            raise

    fastas = list(root_dir.glob('*.fa')) #+ list(sub.glob('*.fasta'))
    logging.info(f"{len(fastas)} files discovered")

    if not fastas:
        raise
    all_records = []
    template = []
    for idx, fa in enumerate(fastas):
        if idx == 0:
            template = read_template(fa)
        all_records.extend(read_last_n(fa, batch_size))


    all_records = [(f">{code}_design_{i}_{iter_number}; {h[18:]}",s) for i,(h,s) in enumerate(all_records)]
    all_records = template + all_records
    out_file = root_dir / f'{code}_merged.fasta'

    with open(out_file, 'w') as f:
        for h, s in all_records:
            f.write(f'{h}\n{s}\n')
    logging.info(f' file has been saved to {out_file} and {len(all_records)} sequences in total')


def gather_results(root_dir):

    ########################################
    # 默认的文件路径是T_0.1_R_20_esm_refined_pdbtm以下就直接是pdbname的目录了，之后就会直接遍历拾取，不需要加别的配置，每次改一下路径和batchsize就行

    #####################################
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(root_dir / d)]
    abs_dirs = [root_dir / d for d in dirs]
    csv_rows = []

    for sub in abs_dirs:
        code = sub.name.lower()
        if not re.fullmatch(r'\w{4}', code):
            continue
        from pdbtm_parser import Pdbtm_parser
        merged_file = sub / f'{code}_merged.fasta'
        header_list = Pdbtm_parser.fasta_utils.extract_all_headers(input_fasta_path=merged_file)
        header_list = header_list[1:] #去掉模板位置
        # print(header_list)

        header_dict = [
                {
                    'name': parts[0].strip(),
                    'overall_identity': float(parts[1].split('=')[1]),
                    'pocket_identity': float(parts[2].split('=')[1]),
                    'chains': parts[3].split('=')[1],  # 不转，保持原样字符串
                    # 'violation': int(parts[4].split('=')[1])
                }
                for h in header_list
                for parts in [h.replace(',chains', ';chains').split(';')]
            ]

        seqs_recovery = [item["overall_identity"] for item in header_dict]
        mean_recovery = st.mean(seqs_recovery) if seqs_recovery else 0.0
        stdev_recovery = st.stdev(seqs_recovery) if len(seqs_recovery) > 1 else 0.0
        csv_rows.append([code, mean_recovery, stdev_recovery])


    import csv
    csv_path = root_dir / "recovery_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["pdbname", "mean_recovery", "stdev_recovery"])
        writer.writerows(csv_rows)

    logging.info(f'CSV has been generated :{csv_path}')

def gather_mpnn_results(root_dir):
    csv_rows = []

    for merged_file in os.listdir(root_dir):
        code = merged_file[:4]
        if not re.fullmatch(r'\w{4}', code):
            continue
        from pdbtm_parser import Pdbtm_parser
        merged_path = Path(os.path.join(root_dir,merged_file))
        header_list = Pdbtm_parser.fasta_utils.extract_all_headers(input_fasta_path=merged_path)
        header_list = header_list[1:] #去掉模板位置
        # print(header_list)

        header_dict = [
                {
                    # 'name': parts[0].strip(),
                    # 'overall_identity': float(parts[1].split('=')[1]),
                    # 'pocket_identity': float(parts[2].split('=')[1]),
                    # 'chains': parts[3].split('=')[1],  # 不转，保持原样字符串
                    # 'violation': int(parts[4].split('=')[1])
                    "seq_recovery" : float(parts[-1].split("=")[-1])

                }
                for h in header_list
                for parts in [h.split(',')]
            ]

        seqs_recovery = [item["seq_recovery"] for item in header_dict]
        mean_recovery = st.mean(seqs_recovery) if seqs_recovery else 0.0
        stdev_recovery = st.stdev(seqs_recovery) if len(seqs_recovery) > 1 else 0.0


        csv_rows.append([code, mean_recovery, stdev_recovery])


    import csv
    csv_path = root_dir / 'recovery_stats.csv'
    with csv_path.open('w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['pdbname', 'mean_recovery', 'stdev_recovery'])
        writer.writerows(csv_rows)

    logging.info(f'CSV has been generated :{csv_path}')


if __name__ == "__main__":
    base = Path('/home/chenty/public_data/third_cluster_data/abacust_design_results/seqs/T_0.1_R_20_esm_refined_zero_afdb_with_0124afdb_cluster_dict_1_3_b48')
    for ckpt in ['checkpoint260', 'checkpoint250', 'checkpoint255']:
        gather_results(base / ckpt)