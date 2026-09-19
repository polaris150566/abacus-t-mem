


import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set

class LeakChecker:
    """MMseqs2 泄露检测封装"""
    def __init__(self, query_fasta: Path, target_fasta: Path, out_dir: str = "leak_check"):
        self.query_fasta = query_fasta
        self.target_fasta = target_fasta
        self.out_dir = Path(out_dir)
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.target_db = self.out_dir / "target_db"
        self.query_db  = self.out_dir / "query_db"
        self.aln_db    = self.out_dir / "aln_db"
        self.tmp_dir   = self.out_dir / "tmp"
        self.hits_tsv  = self.out_dir / "hits.tsv"
        self.report_csv = self.out_dir / "leak_report.csv"
        self.pdb_csv   = self.out_dir / "pdb_leak_summary.csv"

    # ---------- 私有工具 ----------
    def _run(self, cmd: List[str]):
        logging.debug(">>> %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


    def run(self, identity = 0.5, cov = 0.8):
        self._build_db()
        self._search(thres = identity, cov = cov)
        self._convertalis()
        self._analyze()
        self._pdb_summary()
        logging.info("全部完成，报告位于 %s", self.report_csv)
        return self.report_csv

    def _build_db(self) :
        logging.info("1. 创建 MMseqs 数据库")
        self._run(["mmseqs", "createdb", str(self.target_fasta), str(self.target_db)])
        self._run(["mmseqs", "createdb", str(self.query_fasta),  str(self.query_db)])

    def _search(self, thres = 0.5, cov = 0.8) :
        logging.info("2. 运行 search")
        self._run([
            "mmseqs", "search",
            str(self.query_db), str(self.target_db), str(self.aln_db), str(self.tmp_dir),
            "--min-seq-id", f"{thres}",
            "-c", f"{cov}",
            "--cov-mode", "1",
            "--max-seqs", "1000"
        ])

    def _convertalis(self):
        logging.info("3. 生成 hits 列表")
        self._run([
            "mmseqs", "convertalis",
            str(self.query_db), str(self.target_db), str(self.aln_db), str(self.hits_tsv),
            "--format-output",  "query,target,evalue"
        ])

    def _analyze(self):
        logging.info("4. 分析泄露情况")
        leaked = set()
        if self.hits_tsv.exists() and self.hits_tsv.stat().st_size:
            with open(self.hits_tsv) as fh:
                for line in fh:
                    # print(line)
                    leaked.add(line[0:6].lower())
        from protein_utils.fasta_utils import Protein_Sequence

        query_headers = Protein_Sequence.fasta_utils.extract_all_headers(self.query_fasta)
        rows = []
        # print(query_headers)
        # print(leaked)
        for q in query_headers:
            q = q.split(">")[-1].lower()
            rows.append({"query_id": q, "leak_status": "leaked" if q in leaked else "safe"})

        write_csv(self.report_csv, rows)
        total, leaked_cnt = len(rows), len(leaked)
        safe_cnt = total - leaked_cnt
        logging.info("===== 统计 =====")
        logging.info("total_query,%s", total)
        logging.info("leaked,%s", leaked_cnt)
        logging.info("safe,%s", safe_cnt)

    def _pdb_summary(self) :
        logging.info("5. PDB 级别汇总")
        pdb_summary(self.report_csv, self.pdb_csv)

import csv
import logging
from pathlib import Path
from typing import Dict, List


def write_csv(path: Path, rows: List[Dict[str, str]]) :
    if not rows:
        logging.warning("无数据写入 %s", path)
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pdb_summary(report_csv: Path, out_csv: Path) :
    """根据 report_csv 生成 PDB 级别汇总"""
    pdb_map: Dict[str, str] = {}  # pdbid -> "leaked" / "safe"
    with open(report_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q, st = row["query_id"], row["leak_status"]
            pdbid = q[:4]
            if st == "leaked":
                pdb_map[pdbid] = "leaked"
            else:
                pdb_map.setdefault(pdbid, "safe")

    rows = [{"pdb_id": pid, "leak_status": st} for pid, st in pdb_map.items()]
    write_csv(out_csv, rows)

    leak_n = sum(1 for r in rows if r["leak_status"] == "leaked")
    safe_n = len(rows) - leak_n
    logging.info("===== PDB 级别汇总 =====")
    logging.info("leaked_pdb,%s", leak_n)
    logging.info("safe_pdb,%s", safe_n)
    logging.info("PDB 汇总文件：%s", out_csv)

#!/usr/bin/env python3
import logging
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def main():

    checker = LeakChecker(
        query_fasta="/home/chenty/public_data/abacust_mem_data/data_storage/query.fasta" ,
        target_fasta="/home/chenty/public_data/abacust_mem_data/data_storage/target.fasta" ,
        out_dir="/home/chenty/public_data/abacust_mem_data/data_storage/check_leakage/"
        )
    checker.run()

if __name__ == "__main__":
    main()