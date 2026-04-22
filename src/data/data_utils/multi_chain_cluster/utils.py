
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 PDB ID 对应的 FASTA 序列
用法:
    download_raw_chain_from_pdbname('1a1u', './fasta', try_num=3)
"""
import logging
import os
import time
from typing import List
import json
import os
import requests

# 可选：让日志更漂亮
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# RCSB REST API  endpoint
FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"


def download_raw_chain_from_pdbname(pdb_name: str, output_dir: str, try_num: int = 3,overwrite: bool = False,):
    """
    try_num : int
        最大重试次数
    overwrite : bool
        是否覆盖已存在文件
    """
    pdb_id = pdb_name.lower().strip()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB ID 不合法: {pdb_name}")

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{pdb_id}.fasta")

    if not overwrite and os.path.isfile(outfile):
        logging.info("文件已存在，跳过下载: %s", outfile)
        return [outfile]

    session = requests.Session()
    url = FASTA_URL.format(pdb_id=pdb_id)

    for attempt in range(1, try_num + 1):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()

            # RCSB 若返回空字符串，说明 ID 不存在
            if not resp.text.strip():
                logging.warning("RCSB 返回空内容，可能 PDB ID 无效: %s", pdb_id)
                return []

            with open(outfile, "w", encoding="utf-8") as f:
                f.write(resp.text)
            logging.info("已保存 FASTA: %s", outfile)
            return [outfile]

        except requests.RequestException as e:
            logging.warning(
                "第 %d 次下载失败 <%s>: %s", attempt, pdb_id, e
            )
            if attempt < try_num:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                logging.error("最终下载失败 <%s>", pdb_id)
                return []


def read_json(path: str):
    """
    安全地读取 JSON 文件，返回 Python 对象。
    如果文件不存在或 JSON 非法，会抛出 RuntimeError。
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"JSON file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {path}") from e
if __name__ == "__main__":
    download_raw_chain_from_pdbname("1a1u", "./fasta", try_num=3)