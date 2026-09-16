#!/usr/bin/env python3
"""
tmdet 打分管线: 生成的全原子 PDB -> 批量 tmdet(4.1.1, docker) -> qValue / TM边界 / WY环位统计 -> CSV

用法示例:
    python3 /home/chenty/abacust_mem/utils/tmdet_score.py \
        --input_dir /path/to/batch_output/target_dir \
        --out_csv ./tmdet_scores.csv \
        --n_jobs 16 --window 5

行为:
  - 递归扫描 input_dir 下匹配 --pattern 的设计 PDB(默认 *_design_*.pdb);
  - 对每个设计, 在同目录自动找 native 模板(如 {stem}_0.pdb, 即不含 _design_ 的同 stem PDB)
    一并打分, 输出 delta_qvalue = qvalue(设计) - qvalue(native);
  - tmdet 的 xml 结果缓存在 work_dir/xml 下, 重跑时已算过的直接复用(用 --force 强制重算);
  - 依赖: docker + brgenzim/tmdet:4.1.1 镜像。首次运行会联网下载 CCD 到 work_dir/data, 之后复用。

输出 CSV 列:
  file, target, kind, qvalue, tmtype, half_thickness, size, num_tm, tm_boundaries,
  seq_len, global_wy, global_wy_frac, tm_wy, tm_wy_frac,
  ring_len, ring_wy, ring_wy_frac,
  native_qvalue, delta_qvalue, native_ring_wy_frac, delta_ring_wy, error

WY 环位(ring)定义: 距任意 TM 片段边界 ±window 个残基的位置(两端各 window 个),
统计其中 W/Y 的占比。window 用 --window 调整(默认 5)。
"""

import argparse
import concurrent.futures
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TMDET_IMAGE = "brgenzim/tmdet:4.1.1"
TM_TYPES = {"H", "B", "M"}  # H=TM helix, B=TM beta, M=旧版 schema 的膜区


# ----------------------------------------------------------------------------
# 输入收集
# ----------------------------------------------------------------------------
def collect_inputs(input_dir: Path, pattern: str):
    files = sorted(input_dir.rglob(pattern))
    files = [f for f in files if f.suffix.lower() in (".pdb", ".ent") and f.is_file()]
    return files


def classify(p: Path) -> str:
    """native 或 design"""
    return "design" if "_design_" in p.name else "native"


def find_native(design_path: Path) -> Path:
    """对 {stem}_design_*.pdb 找同目录的 native 模板: 优先 {stem}_0.pdb, 其次编号最小的非 design PDB"""
    stem = design_path.name.split("_design_")[0]
    cands = []
    for f in design_path.parent.glob("*.pdb"):
        if "_design_" in f.name:
            continue
        if f.name.startswith(stem + "_"):
            m = re.search(r"_(\d+)\.pdb$", f.name)
            cands.append((int(m.group(1)) if m else 10**9, f))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def _safe_name(rel: Path) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rel))
    if len(s) > 120:
        h = hashlib.md5(str(rel).encode()).hexdigest()[:8]
        s = s[:110] + "_" + h
    return s


# ----------------------------------------------------------------------------
# tmdet 执行
# ----------------------------------------------------------------------------
def run_tmdet(work_dir: Path, ent_rel: Path, xml_rel: Path, timeout=900):
    """docker 跑 tmdet, 返回 (returncode, stderr 摘要)"""
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{work_dir.resolve()}:/work",
        TMDET_IMAGE,
        "-dm", "-fr",
        "-pi", f"/work/{ent_rel.as_posix()}",
        "-po", "",
        "-x", f"/work/{xml_rel.as_posix()}",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    tail = (cp.stderr or "").strip().splitlines()
    return cp.returncode, (tail[-3:] if tail else [])


# ----------------------------------------------------------------------------
# XML 解析与打分
# ----------------------------------------------------------------------------
def parse_xml(xml_path: Path):
    """返回 {qvalue, tmtype, half_thickness, size, chains:[{id, num_tm, seq, tm_regions}]}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for elem in root.iter():  # 剥离默认命名空间, 否则 find 匹配不到
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    raw = root.find("rawData")
    qvalue = float(raw.findtext("qValue")) if raw is not None and raw.findtext("qValue") else None
    tmtype = raw.findtext("tmType") if raw is not None else None

    half_thickness = size = None
    mems = root.find("membranes")
    if mems is not None:
        mem = mems.find("membrane")
        if mem is not None:
            half_thickness = float(mem.get("halfThickness")) if mem.get("halfThickness") else None
            size = float(mem.get("size")) if mem.get("size") else None

    chains = []
    for ch in root.findall("chains/chain"):
        seq = "".join((ch.findtext("sequence") or "").split())
        tm_regions = []
        for r in ch.findall("regions/region"):
            typ = r.get("type", "")
            if typ in TM_TYPES:
                b = int(r.get("startAuthId")); e = int(r.get("endAuthId"))
                tm_regions.append((b, e))
        chains.append({
            "id": ch.get("authId", ""),
            "num_tm": int(ch.get("numTM", 0)),
            "seq": seq,
            "tm_regions": tm_regions,
        })
    return {"qvalue": qvalue, "tmtype": tmtype,
            "half_thickness": half_thickness, "size": size, "chains": chains}


def ring_and_wy_stats(parsed: dict, window: int):
    """环位(±window of TM边界)与全序列/膜区 WY 统计, 多链合并"""
    seq = "".join(c["seq"] for c in parsed["chains"])
    all_tm = []
    offset = 0
    tm_boundaries = []
    for c in parsed["chains"]:
        for b, e in c["tm_regions"]:
            all_tm.append((offset + b, offset + e))
        if len(parsed["chains"]) > 1:
            tm_boundaries.extend(f"{c['id']}:{b}-{e}" for b, e in c["tm_regions"])
        else:
            tm_boundaries.extend(f"{b}-{e}" for b, e in c["tm_regions"])
        offset += len(c["seq"])

    L = len(seq)
    ring_pos = set()
    tm_pos = set()
    for b, e in all_tm:
        for p in range(max(1, b - window), min(L, e + window) + 1):
            ring_pos.add(p)
        for p in range(b, e + 1):
            tm_pos.add(p)

    ring_pos.difference_update(tm_pos)  # 环位 = 边界邻域中不在 TM 段内的位置

    def wy(positions):
        return sum(1 for p in positions if seq[p - 1] in "WY")

    ring_len = len(ring_pos)
    ring_wy = wy(ring_pos)
    tm_len = len(tm_pos)
    tm_wy = wy(tm_pos)
    global_wy = seq.count("W") + seq.count("Y")
    return {
        "seq_len": L,
        "global_wy": global_wy,
        "global_wy_frac": global_wy / L if L else None,
        "tm_wy": tm_wy,
        "tm_wy_frac": tm_wy / tm_len if tm_len else None,
        "ring_len": ring_len,
        "ring_wy": ring_wy,
        "ring_wy_frac": ring_wy / ring_len if ring_len else None,
        "num_tm": sum(c["num_tm"] for c in parsed["chains"]) or len(all_tm),
        "tm_boundaries": ";".join(tm_boundaries),
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="tmdet 批量打分 + WY 环位统计管线")
    ap.add_argument("--input_dir", required=True, help="设计 PDB 所在目录(递归扫描)")
    ap.add_argument("--pattern", default="*_design_*.pdb", help="设计 PDB 的 glob 模式")
    ap.add_argument("--out_csv", default="./tmdet_scores.csv")
    ap.add_argument("--work_dir", default="/home/chenty/tmdet_score_work",
                    help="工作目录(存 input/xml 缓存, CCD 也缓存于此)")
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--window", type=int, default=5, help="WY 环位窗口: TM 边界 ±window 残基")
    ap.add_argument("--force", action="store_true", help="忽略已有 xml, 强制重跑 tmdet")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    ent_dir = work_dir / "input"
    xml_dir = work_dir / "xml"
    ent_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    design_files = collect_inputs(input_dir, args.pattern)
    if not design_files:
        print(f"未找到匹配 {args.pattern} 的 PDB (input_dir={input_dir})")
        sys.exit(1)

    # 收集要打分的所有文件: 设计 + 对应 native
    jobs = {}   # src Path -> xml 相对路径
    native_of = {}
    for d in design_files:
        jobs.setdefault(d, xml_dir / (_safe_name(d.relative_to(input_dir)) + ".xml"))
        n = find_native(d)
        if n is not None:
            native_of[d] = n
            jobs.setdefault(n, xml_dir / (_safe_name(n.relative_to(input_dir)) + ".xml"))

    uniq_natives = {str(n) for n in native_of.values()}
    print(f"共 {len(design_files)} 个设计 PDB, {len(uniq_natives)} 个 native 模板, 总计 {len(jobs)} 个待打分结构")

    # 准备 ent 输入(跳过已有 xml)
    todo = []
    for src, xml_rel in jobs.items():
        if xml_rel.exists() and not args.force:
            continue
        ent_rel = ent_dir / (_safe_name(src.relative_to(input_dir)).rsplit(".", 1)[0] + ".ent")
        shutil.copyfile(src, ent_rel)
        todo.append((src, ent_rel.relative_to(work_dir), xml_rel.relative_to(work_dir)))
    print(f"需运行 tmdet: {len(todo)} 个")

    if todo:
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(run_tmdet, work_dir, ent_rel, xml_rel): src
                    for src, ent_rel, xml_rel in todo}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                src = futs[fut]
                done += 1
                rc, tail = fut.result()
                if rc != 0:
                    errors.append((str(src), tail))
                if done % 20 == 0 or done == len(todo):
                    print(f"  tmdet 进度 {done}/{len(todo)}")
        if errors:
            print(f"  {len(errors)} 个失败(见 CSV 的 error 列), 前3个:")
            for src, tail in errors[:3]:
                print(f"    {src}: {tail}")

    # 解析 + 打分
    rows = []
    for src, xml_rel in jobs.items():
        kind = classify(src)
        row = {
            "file": str(src.relative_to(input_dir)),
            "target": src.name.split("_design_")[0] if kind == "design" else re.sub(r"_\d+$", "", src.stem),
            "kind": kind,
        }
        err = None
        if not xml_rel.exists():
            err = "xml missing (tmdet failed)"
        else:
            try:
                parsed = parse_xml(xml_rel)
                row.update({
                    "qvalue": parsed["qvalue"],
                    "tmtype": parsed["tmtype"],
                    "half_thickness": parsed["half_thickness"],
                    "size": parsed["size"],
                })
                row.update(ring_and_wy_stats(parsed, args.window))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        if err:
            for k in ("qvalue", "tmtype", "half_thickness", "size", "seq_len",
                      "global_wy", "global_wy_frac", "tm_wy", "tm_wy_frac",
                      "ring_len", "ring_wy", "ring_wy_frac", "num_tm", "tm_boundaries"):
                row.setdefault(k, None)
        row["error"] = err
        rows.append(row)

    # native 对照列
    native_rows = {r["file"]: r for r in rows if r["kind"] == "native"}
    for r in rows:
        if r["kind"] != "design":
            continue
        src = input_dir / r["file"]
        n = native_of.get(src)
        nr = native_rows.get(str(n.relative_to(input_dir))) if n is not None else None
        r["native_qvalue"] = nr["qvalue"] if nr else None
        r["delta_qvalue"] = (r["qvalue"] - nr["qvalue"]) if (nr and r["qvalue"] is not None and nr["qvalue"] is not None) else None
        r["native_ring_wy_frac"] = nr["ring_wy_frac"] if nr else None
        r["delta_ring_wy"] = (r["ring_wy_frac"] - nr["ring_wy_frac"]) if (nr and r["ring_wy_frac"] is not None and nr["ring_wy_frac"] is not None) else None

    cols = ["file", "target", "kind", "qvalue", "tmtype", "half_thickness", "size",
            "num_tm", "tm_boundaries", "seq_len",
            "global_wy", "global_wy_frac", "tm_wy", "tm_wy_frac",
            "ring_len", "ring_wy", "ring_wy_frac",
            "native_qvalue", "delta_qvalue", "native_ring_wy_frac", "delta_ring_wy", "error"]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})

    # 汇总
    designs = [r for r in rows if r["kind"] == "design" and not r.get("error")]
    print(f"\n完成: {len(designs)} 个设计打分, CSV -> {out}")
    if designs:
        dq = [r["delta_qvalue"] for r in designs if r["delta_qvalue"] is not None]
        rw = [r["ring_wy_frac"] for r in designs if r["ring_wy_frac"] is not None]
        nw = [r["ring_wy_frac"] for r in rows if r["kind"] == "native" and r["ring_wy_frac"] is not None]
        print(f"  delta_qvalue 均值: {sum(dq)/len(dq):+.2f}" if dq else "  delta_qvalue: 无")
        print(f"  设计 ring_wy_frac 均值: {sum(rw)/len(rw):.3f} (native: {sum(nw)/len(nw):.3f})" if rw and nw else "")


if __name__ == "__main__":
    main()
