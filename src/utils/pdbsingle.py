import torch

# 扩展后的ESM映射表，支持单字母和三字母表示
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


def amino_acid_to_digital_sequence(input_data, mapping = esm_mapping):
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




class PDBinstance:
    '''传入单个pdb文件路径，返回这个pdb文件的解析，提供了一些方法，返回的是这个文件的各种特征，（原子坐标只有主链的）'''
    def __init__(self, file_path):
        """
        初始化 PDBProcessor 类，指定 PDB 文件的路径。
        
        :param file_path: PDB 文件的路径
        """
        self.file_path = file_path
        self.aaindex = [0]
        self.res_type = []
        self.res_atom3 = []
        self.res_atom4 = []
        self.aachain = []
        self.process_pdb_file()

    def process_pdb_file(self):
        """
        处理 PDB 文件，
        """
        with open(self.file_path, 'r') as file:
            aatypetmp = []
            
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
                    
                    # 检查是否进入新的残基
                    if residue_number != self.aaindex[-1]:
                        self.aaindex.append(residue_number)
                        self.res_type.append(residue)
                        self.aachain.append(chain)
                        aatypetmp = []
                    
                    # 处理不同原子名称
                    if atom_name in ['N', 'CA', 'C', 'O']:
                        aatypetmp.append([x_cor, y_cor, z_cor])
                        # 处理特定原子组合
                        if atom_name == 'C' and len(aatypetmp) == 3:
                            self.res_atom3.append(torch.tensor(aatypetmp).reshape(3, 3))
                        elif atom_name == 'O' and len(aatypetmp) == 4:
                            self.res_atom4.append(torch.tensor(aatypetmp).reshape(4, 3))
                            aatypetmp = []
        self.aaindex = self.aaindex[1:]
        self.aaindex = torch.tensor(self.aaindex)
        #print(self.res_type)
        self.res_type = torch.tensor(amino_acid_to_digital_sequence(self.res_type))
        self.res_atom3 = torch.stack(self.res_atom3)
        self.res_atom4 = torch.stack(self.res_atom4)
        
        # 检查是否提取到数据
        if self.aaindex.shape[0] == 0:
            raise ValueError("未从 PDB 文件中提取到任何 ATOM 数据")

    @property
    def full_res_index(self):
        """返回所有残基的索引"""
        return self.aaindex

    @property
    def atom3(self):
        """返回包含 N, CA, C 原子坐标的张量"""
        return self.res_atom3

    @property
    def atom4(self):
        """返回包含 N, CA, C, O 原子坐标的张量"""
        return self.res_atom4

    @property
    def residue_types(self):
        """返回所有残基的类型"""
        return self.res_type

    @property
    def chain_ids(self):
        """返回所有残基的链标识符"""
        return self.aachain

    @property
    def num_residues(self):
        """返回残基数"""
        return len(self.aaindex)