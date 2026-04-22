import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.sequence import ProteinSequence


from biotite.sequence.io import fasta
from biotite.structure.io import pdb as pdb_io

from protein_utils.pdb_parser import Pdb_processer
aatype3to1 = Pdb_processer.infos.AA3_TO_AA1


from joblib import Parallel, delayed
import tqdm


import logging, sys, os
from datetime import datetime
os.makedirs('logs', exist_ok=True)

# 2. 按“年-月-日_时-分”拼日志文件名
log_file = os.path.join('logs', f"{datetime.now():%Y-%m-%d_%H-%M}.log")

# 3. 一次性配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def _residue_chain_sequence(atom_array: struc.AtomArray, chain_id: str) :
    """从 AtomArray 里抽一条链的蛋白序列"""
    chain = atom_array[atom_array.chain_id == chain_id]
    from biotite.structure import filter_amino_acids

    # atom_array 是你的 AtomArray
    aa_atom  = chain[filter_amino_acids(chain)]   # 1. 只留天然氨基酸
    if aa_atom:
        seq = struc.get_residues(aa_atom)[-1]
        # print(seq)
        one_letter = np.vectorize(lambda x: aatype3to1.get(x, 'X'))(seq)
        return   ''.join(one_letter)
    else:
        return ""


import numpy as np
from biotite.structure import AtomArray, get_residue_starts, get_chain_starts

def integrate_insertion_code(atom_array: AtomArray) :
    # import pdb;pdb.set_trace()
    arr          = atom_array.copy()
    residue_idx  = get_residue_starts(arr)          # 残基首原子索引
    chain_idx    = get_chain_starts(arr)            # 链首原子索引
    mask = np.zeros(len(atom_array.res_id), dtype=bool)      # 全 0 False
    mask[residue_idx] = True

    # 1. 链起止切片
    chain_bounds = [(chain_idx[i], chain_idx[i+1]-1) for i in range(len(chain_idx)-1)]
    chain_bounds.append((chain_idx[-1], arr.array_length()-1))

    total_offset = []
    # import pdb;pdb.set_trace()
    for start, stop in chain_bounds:
        sub_ins_array = arr.ins_code[start:stop+1]
        ins_flag = (sub_ins_array != '' ) & mask[start:stop+1]      # 残基级布尔
        offset   = np.cumsum(ins_flag)              # 累加偏移
        total_offset.append(offset)

    arr.res_id = np.concatenate(total_offset, axis=0) + arr.res_id
    # 4. 清插入码
    arr.ins_code[:] = ""
    return arr

def _sequence_identity(seq1: ProteinSequence, seq2: ProteinSequence) :
    """简单全局 identity"""
    if len(seq1) == 0 or len(seq2) == 0:
        return False
    return seq1 == seq2

def _representative_chains_by_sequence(atom_array: struc.AtomArray) :
    """对着atomarray进行链层级额去冗余"""
    chain_ids = sorted(set(atom_array.chain_id))
    seen_seq = {}
    for cid in chain_ids:
        seq = _residue_chain_sequence(atom_array, cid)
        if seq and seq not in seen_seq:
            seen_seq[seq] = cid

    return list(seen_seq.values())

def _process_one_pdb(pdb_path: Path, out_dir: Path, fasta_dir: Path, out_fmt: str) :
    try:

        pdb_file = pdb_io.PDBFile.read(str(pdb_path))
        pdb_id = pdb_path.stem.lower()[:4]
        print(pdb_id)

        ################################################################
        # 分辨率
        header = strucio.pdb.PDBFile.read(str(pdb_path)).get_remark(2)
        reso = 0.0
        for line in header:
            if "RESOLUTION" in line and "ANGSTROMS" in line:
                try:
                    reso = float(line.split("RESOLUTION", 1)[-1][1:].split("ANGSTROMS")[0].strip())
                except:
                    logger.warning(f"{pdb_path} get resolution is {line.split('RESOLUTION', 1)[-1][1:].split('ANGSTROMS')[0].strip()}, please check !")
                break

        reso_str = f"{int(reso*100):03d}" if reso > 0 else "000"



        #########################################################
        # 组装 assembly
        avail_ass = pdb_io.list_assemblies(pdb_file)
        if avail_ass:
            assembly_id = avail_ass[0]           # 默认用第一个
            assembly_atom_array = pdb_io.get_assembly(
                pdb_file,
                assembly_id=assembly_id,
                model=1,            # 保持单模型
                altloc='occupancy',
                extra_fields=['atom_id',"b_factor",'occupancy']
            )
        else:
            logger.warning(f"pdb {pdb_path} has no biomatrix, please check !")
            assembly_atom_array = pdb_file.get_structure(model=1)

        ##################################################
        # insertion code
        has_ins = np.any(assembly_atom_array.ins_code != "")
        if has_ins:
            logging.info(f"{pdb_path} has insertion code")
            assembly_atom_array = integrate_insertion_code(assembly_atom_array)

        ####################################################################
        # 判断 sigl/homo/hetr
        chain_ids = sorted(set(assembly_atom_array.chain_id))
        if len(chain_ids) == 1:
            olig_type = "sigl"
        else:
            # 比较所有链序列
            seqs = [_residue_chain_sequence(assembly_atom_array, cid) for cid in chain_ids]
            seqs = [ seq for seq in seqs if seq]
            identical = all(_sequence_identity(seqs[0], s) == True for s in seqs[1:])
            olig_type = "homo" if identical else "hetr"

        # ----- 5. 输出文件名 -----
        out_name = f"{pdb_id}_{reso_str}_{olig_type}_.{out_fmt}"
        out_path = out_dir / out_name
        tmp_path = out_path.with_suffix('.pdb')
        strucio.save_structure(str(tmp_path), assembly_atom_array)
        # biotite 识别不了ent
        tmp_path.rename(out_path)

        ###################################################
        # 抽取fasta
        if olig_type in ("sigl", "homo"):
            rep_chains = [chain_ids[0]]
        else:
            rep_chains = _representative_chains_by_sequence(assembly_atom_array)

        seq_records = []
        for cid in rep_chains:
            # import pdb; pdb.set_trace()
            seq = _residue_chain_sequence(assembly_atom_array, cid)
            if seq:                                      #注意，这里有可能是整个一条链全是hetatm，这时得到的序列就是全空的
                seq_records.append(f">{pdb_id}_{cid}\n{seq}\n")
        if seq_records:
            (fasta_dir / f"{pdb_id}.fasta").write_text("".join(seq_records))

    except Exception as e:
        logger.exception(f"Failed on {pdb_path}: {e}")

# ---------- 主入口 ----------
def transform_and_annotate_pdb_files(
    input_dir: str = "",
    output_dir: str = "",
    fasta_output_dir: str = "",
    output_format: str = "ent",
    n_jobs: int = -1,
):
    """
    并行遍历 input_dir 下所有 *.pdb / *.ent，完成：
    1. 组装 biological assembly（REMARK 350）；
    2. 按规则重命名并写回 output_dir；
    3. 抽取 FASTA（单链/同源只留一条，异源去冗余）到 fasta_output_dir。
    文件名模板：
        pdbId_分辨率(3位)_{sigl|homo|hetr}_{ins|cln}.ent
    """
    logger.info("Begin....")
    if not input_dir or not output_dir:
        logger.error("input_dir and output_dir must be given!")
        return

    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    fasta_path = Path(fasta_output_dir) if fasta_output_dir else out_path / "fasta"
    out_path.mkdir(parents=True, exist_ok=True)
    fasta_path.mkdir(parents=True, exist_ok=True)

    assert output_format in {"ent", "pdb", "pdbqt"}
    # 收集所有 pdb/ent
    # files = list(in_path.rglob("*.pdb")) + list(in_path.rglob("*.ent"))
    ###################################################################################################################################################################################
    keep_prefixes = {'1brd', '7a5v', '1qo1', "1q01", "6hda","6hdb","6hd9","6hd8","6hdc","2vpz","40xx", "4oxx", "9cdc","3lnm","1w5c","2a06","6swr","4y7k","5era","6w0f","7oph"}   # 针对指定的pdb重新生成

    files = [p for p in in_path.rglob('*.pdb') if p.stem[:4].lower() in keep_prefixes] + \
            [p for p in in_path.rglob('*.ent') if p.stem[:4].lower() in keep_prefixes]
    #############################################################################################################################################################################
    if not files:
        logger.warning("No pdb/ent found in input_dir!")
        return

    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_one_pdb)(f, out_path, fasta_path, output_format)
        for f in tqdm.tqdm(files, desc="Processing")
    )
    logger.info("All done.")


if __name__ == "__main__":
    try:
        transform_and_annotate_pdb_files(input_dir = "../raw_data/pdbs",output_dir = "../data_storage/assembled_pdbs",fasta_output_dir="../data_storage/fasta_for_cluster",
                    output_format="ent",
                    n_jobs=1)
    except Exception as e:
        logger.exception(f"error: {e}")
        raise
