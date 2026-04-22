import torch
from pdbsingle import PDBinstance
import os

class PDBFileHandler:
    def __init__(self, *pdb_paths):
        self.pdb_paths = pdb_paths
        self.data = [PDBinstance(path) for path in pdb_paths]

    @classmethod
    def fromfilenames(cls, file_path):
        """从文件中读取PDB文件路径列表，并创建PDBinstance对象"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件未找到: {file_path}")
            return None
        
        # 检查是否是文件
        if not os.path.isfile(file_path):
            print(f"无效的文件名: {file_path}")
            return None
        
        # 读取文件内容并创建PDBinstance对象
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if line:  # 忽略空行
                        data.append(PDBinstance(line))
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return None
        
        # 创建类的实例
        instance = cls()
        instance.data = data
        return instance

    @property
    def data_tensor(self):
        """返回一个字典，包括aaindex,restype,resatom3,resatom4,(B,L)(B,L),(B,L,3,3),(B,L,4,3)"""
        # 检查data是否为空
        if not self.data:
            raise ValueError("data列表为空")
        
        # 提取每个PDBinstance对象的属性并堆叠
        aaindex_list = [item.aaindex for item in self.data]
        res_type_list = [item.res_type for item in self.data]
        res_atom3_list = [item.res_atom3 for item in self.data]
        res_atom4_list = [item.res_atom4 for item in self.data]
        
        # 在新维度上堆叠张量
        aaindex = torch.stack(aaindex_list, dim=0)
        res_type = torch.stack(res_type_list, dim=0)
        res_atom3 = torch.stack(res_atom3_list, dim=0)
        res_atom4 = torch.stack(res_atom4_list, dim=0)
        
        return {
            'aaindex': aaindex,
            'res_type': res_type,
            'res_atom3': res_atom3,
            'res_atom4': res_atom4
        }
    @property
    def batch_size(self):
        return self.data.length()
    
    