from rdkit import Chem
import logging
import os
import sys
from rdkit.Chem import AllChem
import subprocess

sys.path.append('/home/chenty/abacust_mem/src/utils/')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_file(input_file, output_file):
    """
    使用 RDKit 将化学文件在 PDB、MOL2、SDF 和 PDBQT 格式之间进行转换。

    Args:
        input_file (str): 输入文件的路径。
        output_file (str): 输出文件的路径。

    Returns:
        bool: 转换成功返回 True，失败返回 False。
    """
    try:
        # 获取输入和输出文件的扩展名
        input_ext = os.path.splitext(input_file)[1][1:].lower()
        output_ext = os.path.splitext(output_file)[1][1:].lower()

        subprocess.run(["/home/chenty/miniconda3/envs/abacust/bin/obabel", input_file,"-O", output_file])
        return True
    except Exception as e:
        logging.error(f"转换失败: {e}")
        return False



# 示例用法
if __name__ == "__main__":
    print("testing...")

    input_file = "/home/chenty/abacust_mem/src/data/pdbs/1a0s.pdb"
    output_file = '/home/chenty/abacust_mem/src/data/npy_gen/example.mol2'

    success = convert_file(input_file, output_file)
    print("文件转换成功！" if success else "文件转换失败！")