"""
pymol_mem_vis.py
----------------
工具脚本：根据PDBTM仿射变换矩阵（存储于.npy文件），在PyMOL中
将跨膜区残基染蓝色（cartoon），并以红色（W）/绿色（Y）显示侧链。

用法：
    python pymol_mem_vis.py \
        --pdb-glob "/path/to/seqs/*/**/*_design_*_19.pdb" \
        --npy-dir  /path/to/npys_tmdet \
        --out-dir  /path/to/output \
        [--prot-field 0]   # 蛋白质目录名按_分割后，第几个字段是4字符PDB code（默认0）

输出：
    <out-dir>/<label>.pdb  — B因子替换为膜掩码的PDB（1.0=膜内，0.0=膜外）
    <out-dir>/vis.py       — PyMOL批量脚本，用 `pymol -c vis.py` 运行
"""

import sys, os, glob, argparse
import numpy as np

sys.path.insert(0, "/home/chenty/protein_utils")
from pdbtm_data_parser import Pdbtm_parser


# ---------------------------------------------------------------------------
# 几何核心
# ---------------------------------------------------------------------------

def mask_residues(pdb_path, npy_path):
    """计算每个Cα残基是否在膜内，返回 (chain, resnum, b值) 列表和半厚度。"""
    p = Pdbtm_parser(npy_path)
    R    = p._tmatrix[:, :3]   # (3,3) 旋转矩阵
    T    = p._tmatrix[:, 3]    # (3,)  平移向量
    half = p._membrane_halfthickness

    res = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") and line[12:16].strip() == "CA"):
                continue
            ch = line[21]
            rn = int(line[22:26].strip())
            ca = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            r_mem = R @ (ca + T)   # 先平移再旋转，z轴为膜法向
            bv = 1.0 if abs(r_mem[2]) < half else 0.0
            res.append((ch, rn, bv))
    return res, half


def write_bfactor_pdb(src_pdb, residues, out_pdb):
    """将PDB的B因子列替换为膜掩码值（1.0/0.0），写出新PDB。"""
    md = {(r[0], r[1]): r[2] for r in residues}
    with open(src_pdb) as fi, open(out_pdb, "w") as fo:
        for line in fi:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    k  = (line[21], int(line[22:26].strip()))
                    bv = md.get(k, 0.0)
                    line = line[:60] + "%6.2f" % bv + line[66:]
                except Exception:
                    pass
            fo.write(line)


# ---------------------------------------------------------------------------
# PyMOL脚本生成
# ---------------------------------------------------------------------------

def pymol_commands_for(label, out_pdb, png_path):
    """生成单个结构的PyMOL命令列表。"""
    return [
        "cmd.load('%s', '%s')"          % (out_pdb, label),
        "cmd.show_as('cartoon', '%s')"  % label,
        "cmd.color('white', '%s')"      % label,
        # 膜内区域染蓝
        "cmd.select('m_%s', '%s and b > 0.5')" % (label, label),
        "cmd.color('blue', 'm_%s')"     % label,
        # W侧链红色
        "cmd.show('sticks', '%s and resn TRP')" % label,
        "cmd.color('red',   '%s and resn TRP')" % label,
        # Y侧链绿色
        "cmd.show('sticks', '%s and resn TYR')" % label,
        "cmd.color('green', '%s and resn TYR')" % label,
        # 渲染导出
        "cmd.bg_color('white')",
        "cmd.set('ray_opaque_background', 1)",
        "cmd.orient('%s')" % label,
        "cmd.ray(800, 600)",
        "cmd.png('%s')"    % png_path,
        "cmd.delete('all')",
    ]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(pdb_glob, npy_dir, out_dir, prot_field=0):
    os.makedirs(out_dir, exist_ok=True)

    pdbs = sorted(glob.glob(pdb_glob, recursive=True))
    if not pdbs:
        print("未找到PDB文件:", pdb_glob)
        sys.exit(1)
    print(f"共找到 {len(pdbs)} 个PDB文件")

    pymol_lines = ["from pymol import cmd"]
    skipped = 0

    for pdb_path in pdbs:
        fname = os.path.basename(pdb_path)
        prot  = os.path.basename(os.path.dirname(pdb_path))  # 父目录名即蛋白质名
        # 取4字符PDB code用于查找npy
        code4 = prot.split("_")[prot_field][:4]
        # 先按完整蛋白质名查找，找不到再按4字符前缀匹配
        npy_path = os.path.join(npy_dir, prot + ".npy")
        if not os.path.exists(npy_path):
            candidates = glob.glob(os.path.join(npy_dir, code4 + "*.npy"))
            npy_path   = candidates[0] if candidates else None

        if npy_path is None or not os.path.exists(npy_path):
            print(f"SKIP {prot}: 在 {npy_dir} 中未找到对应npy")
            skipped += 1
            continue

        label    = prot + "__" + os.path.splitext(fname)[0].split("_design_")[-1]
        out_pdb  = os.path.join(out_dir, label + ".pdb")
        png_path = os.path.join(out_dir, label + ".png")

        try:
            residues, half = mask_residues(pdb_path, npy_path)
        except Exception as e:
            print(f"ERROR {prot}: {e}")
            skipped += 1
            continue

        n_mem = sum(1 for r in residues if r[2] > 0.5)
        print(f"{prot} | 半厚度={half:.1f} | 膜内={n_mem}/{len(residues)}")
        write_bfactor_pdb(pdb_path, residues, out_pdb)
        pymol_lines += pymol_commands_for(label, out_pdb, png_path)

    vis_path = os.path.join(out_dir, "vis.py")
    with open(vis_path, "w") as f:
        f.write("\n".join(pymol_lines) + "\n")

    total = len(pdbs) - skipped
    print(f"\n完成：{total} 个结构已写出，{skipped} 个跳过")
    print(f"PyMOL脚本：{vis_path}")
    print(f"运行方式：  pymol -c {vis_path}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-glob",   required=True,
                    help="设计PDB文件的glob模式（记得加引号）")
    ap.add_argument("--npy-dir",    required=True,
                    help="存放.npy变换矩阵文件的目录")
    ap.add_argument("--out-dir",    required=True,
                    help="输出目录（B因子PDB和vis.py）")
    ap.add_argument("--prot-field", type=int, default=0,
                    help="蛋白质目录名按_分割后第几个字段是4字符PDB code（默认0）")
    args = ap.parse_args()
    run(args.pdb_glob, args.npy_dir, args.out_dir, args.prot_field)
