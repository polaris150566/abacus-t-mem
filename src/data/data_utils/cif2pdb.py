#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, Select
# from Bio.PDB.StructureBuilder import PDBConstructionWarning
from joblib import Parallel, delayed
from typing import List, Tuple
import warnings
from tqdm import tqdm

# 忽略 Biopython 的残基构造警告
# warnings.simplefilter('ignore', PDBConstructionWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('cif2pdb.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)



class _AgFilter(Select):
    """去掉 AG 原子"""
    def accept_atom(self, atom):
        return atom.name.strip() != "AG"


class _NoFilter(Select):
    """全保留"""
    def accept_atom(self, atom):
        return True

class Cif2PdbConverter:
    """CIF → PDB 转换器，可选剔除 AG 原子"""

    @staticmethod
    def convert(cif_path: str,
                pdb_path: str,
                remove_ag: bool = True,
                delete_src: bool = False):
        """
        单文件转换
        :param cif_path: 输入 .cif 文件路径
        :param pdb_path: 输出 .pdb 文件路径（含文件名）
        :param remove_ag: 是否剔除银离子（AG）
        :param delete_src: 是否删除原 cif 文件
        :return: 成功 True，失败 False
        """
        try:
            structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
            io = PDBIO()
            io.set_structure(structure)

            # 自动建目录
            Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)

            selector = _AgFilter() if remove_ag else _NoFilter()
            io.save(pdb_path, select=selector)

            if delete_src:
                Path(cif_path).unlink()
            log.info(f"[OK]  {cif_path}  ->  {pdb_path}  (AG removed: {remove_ag})")
            return True

        except Exception as e:
            log.error(f"[ERR] {cif_path} : {e}")
            return False

    @staticmethod
    def batch_convert(root_dir: str,output_dir: str,remove_ag: bool = True, delete_src: bool = False, n_jobs: int = -1):
        """
        并行转换 root_dir 下所有 *.tr.cif
      """
        root_path = Path(root_dir)
        out_path  = Path(output_dir)

        # 预先生成 (cif, pdb) 列表
        tasks: List[Tuple[str, str]] = [(str(cif_file), str(out_path / cif_file.relative_to(root_path).with_suffix('.pdb')))
                                        for cif_file in root_path.rglob('*.tr.cif')]
        if not tasks:
            log.warning("no any *.tr.cif ")
            return

        log.info("begin converting, %d files in total，n_jobs=%s", len(tasks), n_jobs)

        # joblib 并行
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(Cif2PdbConverter.convert)(
                cif, pdb, remove_ag=remove_ag, delete_src=delete_src
            )
            for cif, pdb in tqdm(tasks)
        )

        ok_cnt = sum(results)
        log.info("converting success : %d / %d ", ok_cnt, len(tasks))

if __name__ == '__main__':
    # 1. 单文件
    # Cif2PdbConverter.convert(
    #     cif_path="/tmp/1abc.tr.cif",
    #     pdb_path="/output/1abc_noAG.pdb",
    #     remove_ag=True,
    #     delete_src=False
    # )

    # 2. 批量目录
    Cif2PdbConverter.batch_convert(
        root_dir="/home/chenty/public_data/tmdet_data/out",
        output_dir="/home/chenty/public_data/tmdet_data/pdbs",
        remove_ag=True,
        delete_src=False,
        n_jobs=128
    )