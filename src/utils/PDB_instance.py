import os
import torch
from typing import Dict, List, Tuple, Union


esm_mapping = {
    # 单字母表示
    'A': 0,  # Alanine
    'R': 1,  # Arginine
    'N': 2,  # Asparagine
    'D': 3,  # Aspartic acid
    'C': 4,  # Cysteine
    'Q': 5,  # Glutamine
    'E': 6,  # Glutamic acid
    'G': 7,  # Glycine
    'H': 8,  # Histidine
    'I': 9,  # Isoleucine
    'L': 10, # Leucine
    'K': 11, # Lysine
    'M': 12, # Methionine
    'F': 13, # Phenylalanine
    'P': 14, # Proline
    'S': 15, # Serine
    'T': 16, # Threonine
    'W': 17, # Tryptophan
    'Y': 18, # Tyrosine
    'V': 19, # Valine
    
    # 三字母表示
    'ALA': 0,  # Alanine
    'ARG': 1,  # Arginine
    'ASN': 2,  # Asparagine
    'ASP': 3,  # Aspartic acid
    'CYS': 4,  # Cysteine
    'GLN': 5,  # Glutamine
    'GLU': 6,  # Glutamic acid
    'GLY': 7,  # Glycine
    'HIS': 8,  # Histidine
    'ILE': 9,  # Isoleucine
    'LEU': 10, # Leucine
    'LYS': 11, # Lysine
    'MET': 12, # Methionine
    'PHE': 13, # Phenylalanine
    'PRO': 14, # Proline
    'SER': 15, # Serine
    'THR': 16, # Threonine
    'TRP': 17, # Tryptophan
    'TYR': 18, # Tyrosine
    'VAL': 19  # Valine
}

def amino_acid_to_digital_sequence(input_data, mapping=esm_mapping):
    """
    将氨基酸序列映射为数字序列，并展开成一维数组

    参数:
        input_data (str, list): 氨基酸序列，可以是单个字符、字符串或字符数组（支持嵌套数组）
        mapping (dict): 氨基酸到数字的映射表

    返回:
        list: 一维数字序列
    """
    result = []
    
    def recursive_process(data):
        if isinstance(data, str):
            # 如果是字符串，逐字符或逐三字母处理
            if len(data) == 1:
                # 单字母表示
                aa = data.upper()
                if aa in mapping:
                    result.append(mapping[aa])
                else:
                    raise ValueError(f"Unknown amino acid symbol: {data}")
            else:
                # 三字母表示，按步长3处理
                for i in range(0, len(data), 3):
                    triplet = data[i:i+3].upper()
                    if triplet in mapping:
                        result.append(mapping[triplet])
                    else:
                        raise ValueError(f"Unknown amino acid symbol: {triplet}")
        elif isinstance(data, list):
            # 如果是列表，递归处理每个元素
            for item in data:
                if isinstance(item, str):
                    if len(item) == 1 or len(item) == 3:
                        recursive_process(item)
                    else:
                        raise ValueError("String elements must be single-character or three-character strings")
                elif isinstance(item, list):
                    recursive_process(item)
                else:
                    raise ValueError("List elements must be strings or lists")
        else:
            # 不支持的类型
            raise TypeError("Input data must be a string or a list")
    
    recursive_process(input_data)
    return [int(x) for x in result]

class PDBsingle:
    '''传入单个pdb文件路径，返回这个pdb文件的解析，提供了一些方法，返回的是这个文件的各种特征，（原子坐标只有主链的）'''
    def __init__(self, file_path):
        """
        初始化 PDBsingle 类，指定 PDB 文件的路径。
        
        :param file_path: PDB 文件的路径
        """
        self.file_path = file_path
        self.chains = {}  # 按链存储数据
        self.process_pdb_file()
        
    @classmethod
    def from_pdb_string(cls, pdb_string: str, base_path=None):
        """
        从PDB文件的字符串内容创建 PDBsingle 实例
        
        :param pdb_string: 四位数pdb标号
        :return: PDBsingle 实例
        """
        if not base_path:
            base_path = '/home/chenty/fine-tuning-for-tmpr/src/data/pdbs/pdbs_v/'
        pdb_file = f"{pdb_string}.pdb"
        full_path = os.path.join(base_path, pdb_file)
        return cls(full_path)
    
    def process_pdb_file(self):
        """
        处理 PDB 文件，按链分组解析数据
        """
        current_chain = None
        current_residue = None
        residue_atoms = []

        with open(self.file_path, 'r') as file:
            for line in file:
                if line.startswith('ATOM'):
                    # 提取每列的数据
                    atom_name = line[12:16].strip()
                    residue_number = int(line[22:26].strip())
                    residue = line[17:20].strip()
                    chain = line[21].strip()
                    x_cor = float(line[30:38].strip())
                    y_cor = float(line[38:46].strip())
                    z_cor = float(line[46:54].strip())

                    # 检查是否进入新的链
                    if chain != current_chain:
                        if current_chain is not None:
                            # 处理当前链的最后一个残基
                            self._process_residue(current_chain, current_residue, residue_atoms)
                        current_chain = chain
                        current_residue = residue_number
                        residue_atoms = []
                        self._initialize_chain(chain)

                    # 检查是否进入新的残基
                    if residue_number != current_residue:
                        self._process_residue(current_chain, current_residue, residue_atoms)
                        current_residue = residue_number
                        residue_atoms = []

                    # 处理主链原子
                    if atom_name in ['N', 'CA', 'C', 'O']:
                        residue_atoms.append((atom_name, [x_cor, y_cor, z_cor]))

            # 处理文件最后一行
            if residue_atoms:
                self._process_residue(current_chain, current_residue, residue_atoms)

        # 检查是否提取到数据
        if not self.chains:
            raise ValueError("未从 PDB 文件中提取到任何 ATOM 数据")

    def _initialize_chain(self, chain_id: str):
        """
        初始化一个新链的数据结构
        """
        self.chains[chain_id] = {
            'aaindex': [],
            'res_type': [],
            'res_atom3': [],
            'res_atom4': [],
            'aachain': chain_id
        }

    def _process_residue(self, chain_id: str, residue_number: int, atoms):
        """
        处理一个残基的数据
        """
        if not atoms:
            return

        # 提取残基类型
        residue_type = None
        for atom in atoms:
            if atom[0] == 'CA':
                residue_type = self._get_residue_type(chain_id, atoms)
                break

        if residue_type is None:
            return

        # 更新链数据
        chain_data = self.chains[chain_id]
        chain_data['aaindex'].append(residue_number)
        chain_data['res_type'].append(residue_type)

        # 处理原子坐标
        atom_coords = []
        for atom in atoms:
            if atom[0] in ['N', 'CA', 'C', 'O']:
                atom_coords.append(atom[1])

        if len(atom_coords) >= 3:
            chain_data['res_atom3'].append(torch.tensor(atom_coords[:3]).reshape(3, 3))
        if len(atom_coords) >= 4:
            chain_data['res_atom4'].append(torch.tensor(atom_coords[:4]).reshape(4, 3))


    def detect_gaps(self, chain_id: str):
        """
        检测指定链中的残基序号不连续的点，并返回 gap 信息

        :param chain_id: 链的标识符
        :return: 一个列表，包含 gap 的起始和结束序号
        """
        chain_data = self.chains[chain_id]
        residue_indices = chain_data['aaindex']
        
        gaps = []
        for i in range(1, len(residue_indices)):
            current_idx = residue_indices[i]
            prev_idx = residue_indices[i-1]
            
            if current_idx != prev_idx + 1:
                gaps.append((prev_idx, current_idx))
        
        return gaps

    def get_continuous_segments(self, chain_id: str):
        """
        根据 gap 信息返回氨基酸序列的连续段落标注

        :param chain_id: 链的标识符
        :return: 一个列表，每个元素是一个元组，表示连续段的起始和结束序号
        """
        chain_data = self.chains[chain_id]
        residue_indices = chain_data['aaindex']
        gaps = self.detect_gaps(chain_id)
        
        segments = []
        start_idx = residue_indices[0]
        
        for gap in gaps:
            end_idx = gap[0]
            segments.append((start_idx, end_idx))
            start_idx = gap[1]
        
        # 添加最后一个段落
        segments.append((start_idx, residue_indices[-1]))
        
        return segments
    @property
    def chain_ids(self):
        """返回所有链的标识符"""
        return list(self.chains.keys())

    @property
    def num_residues(self):
        """返回每个链的残基数"""
        return {chain: len(data['aaindex']) for chain, data in self.chains.items()}

    def get_chain_data(self, chain_id: str):
        """
        获取指定链的数据
        """
        if chain_id not in self.chains:
            raise ValueError(f"链 {chain_id} 不存在")
        chain_data = self.chains[chain_id]
        return {
            'aaindex': torch.tensor(chain_data['aaindex']),
            'res_type': torch.tensor(chain_data['res_type']),
            'res_atom3': torch.stack(chain_data['res_atom3']),
            'res_atom4': torch.stack(chain_data['res_atom4']),
            'aachain': chain_id
        }

class PDB_instance:
    def __init__(self, pdb_ids, _path: str):
        """
        初始化PDB_instance类
        参数:
            pdb_ids: PDB文件ID列表,里面装的是数组。
            _path: PDB文件的本地路径，所有所需的pdb文件都要装在这个文件夹里
        """
        self.pdb_ids = pdb_ids
        self._path = _path
        self.content = {}  # 储存的是每一个pdb文件解析后的结果
        
        for _id in pdb_ids:
            content = PDBsingle.from_pdb_string(_id, base_path=self._path)
            self.content[_id] = content

    @classmethod
    def from_pdb(cls, pdb_list, _path=None):
        """
        从PDB文件的字符串内容创建 PDBsingle 实例
        
        :param pdb_list: 四位数pdb标号或者是相应的字典或者元组
        :return: PDBinstance 实例
        """
        if not _path:
            _path = '/home/chenty/fine-tuning-for-tmpr/src/data/pdbs/pdbs_v/'
        if isinstance(pdb_list, (list, tuple)):
            pdb_list = list(pdb_list)
        elif isinstance(pdb_list, str):
            pdb_list = [pdb_list]
        else:
            raise TypeError("输入必须是字符串、列表或元组")
        return cls(pdb_list, _path)

    def get_pdb_data(self, pdb_id: str) :
        """
        根据给出的pdbid字符串在类中查找，返回的是一个pdbsingle类
        
        :param pdb_id: PDB 文件的 ID
        :return: PDBsingle 实例
        """
        if pdb_id not in self.content:
            raise ValueError(f"PDB ID {pdb_id} 不存在")
        return self.content[pdb_id]

    def get_atom3(self, pdb_id: str, chain_id: str) :
        """
        获取指定 PDB ID 和链的 N, CA, C 原子坐标
        
        :param pdb_id: PDB 文件的 ID
        :param chain_id: 链的标识符
        :return: 形状为 (L, 3, 3) 的张量
        """
        pdb_data = self.get_pdb_data(pdb_id)
        return pdb_data.get_chain_data(chain_id)['res_atom3']

    def get_atom4(self, pdb_id: str, chain_id: str):
        """
        获取指定 PDB ID 和链的 N, CA, C, O 原子坐标
        
        :param pdb_id: PDB 文件的 ID
        :param chain_id: 链的标识符
        :return: 形状为 (L, 4, 3) 的张量
        """
        pdb_data = self.get_pdb_data(pdb_id)
        return pdb_data.get_chain_data(chain_id)['res_atom4']

    def get_residue_types(self, pdb_id: str, chain_id: str) :
        """
        获取指定 PDB ID 和链的残基类型
        
        :param pdb_id: PDB 文件的 ID
        :param chain_id: 链的标识符
        :return: 一维张量，包含残基类型的数字编码
        """
        pdb_data = self.get_pdb_data(pdb_id)
        return pdb_data.get_chain_data(chain_id)['res_type']

    def get_chain_ids(self, pdb_id: str):
        """
        获取指定 PDB ID 的所有链标识符
        
        :param pdb_id: PDB 文件的 ID
        :return: 链标识符列表
        """
        pdb_data = self.get_pdb_data(pdb_id)
        return pdb_data.chain_ids

    def get_num_residues(self, pdb_id: str, chain_id: str):
        """
        获取指定 PDB ID 和链的残基数
        
        :param pdb_id: PDB 文件的 ID
        :param chain_id: 链的标识符
        :return: 残基数
        """
        pdb_data = self.get_pdb_data(pdb_id)
        return len(pdb_data.get_chain_data(chain_id)['aaindex'])

    def apply_transformation(self, 
                             tensor: torch.Tensor, 
                             rotation_matrix: torch.Tensor, 
                             translation_vector: torch.Tensor, 
                             scaling_factor: float = 1.0
                             ):
        """
        对张量中的每个原子坐标应用仿射变换
        
        :param tensor: 形状为 (L, 4, 3) 的张量
        :param rotation_matrix: 3x3 旋转矩阵
        :param translation_vector: 3维平移向量
        :param scaling_factor: 缩放因子 (可选)
        :return: 应用变换后的张量
        """
        # 确保输入是张量
        tensor = torch.tensor(tensor, dtype=torch.float32)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
        translation_vector = torch.tensor(translation_vector, dtype=torch.float32)

        # 应用缩放
        scaled_tensor = tensor * scaling_factor

        # 应用旋转
        rotated_tensor = torch.einsum('ijk,kl->ijl', scaled_tensor, rotation_matrix)

        # 应用平移
        transformed_tensor = rotated_tensor + translation_vector

        return transformed_tensor

