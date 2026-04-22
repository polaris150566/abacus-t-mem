
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
import csv
from pathlib import Path
from typing import List
import json
import os
import requests
import time
# 可选：让日志更漂亮
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def read_csv_to_dict(file_path,  delimiter: str = ',', encoding: str = 'utf-8') :
    """
    读取 CSV 文件，返回字典：
        {行号(从0开始): [列1, 列2, ...]}
    忽略表头（如果有，就当普通行处理）。
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f'CSV 文件不存在: {path}')

    with path.open(newline='', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=delimiter)
        return {idx: row for idx, row in enumerate(reader)}

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

def save_dict_as_json(file_path,data, indent = 4, ensure_ascii = False) :
    """
    将字典保存为 JSON 文件（UTF-8，缩进 4）
    参数
    ----
    data        : dict
        要保存的字典
    file_path   : str | pathlib.Path
        输出文件路径（若目录不存在会自动创建）
    indent      : int, 默认 4
        JSON 缩进
    ensure_ascii: bool, 默认 False
        是否强制转义非 ASCII 字符（False 可保留中文）
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)