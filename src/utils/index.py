import torch
import torch.nn as nn

from pdbdataset import PDBFileHandler, PDBinstance
from mask import Mask_Processor
from distogram import Distogram

class PDBD:
    def __init__(self, file_list_path, num_bins=31, min_dist=2.0, max_dist=20.0):
        """
        初始化PDBD类，加载文件列表和相关数据。
        
        参数:
            file_list_path (str): 包含PDB文件路径的文件列表。
            num_bins (int): Distogram的bin数量。
            min_dist (float): Distogram的最小距离。
            max_dist (float): Distogram的最大距离。
        """
        self.file_list_path = file_list_path
        self.mask_processor = Mask_Processor()  # 加载掩码处理器
        self.distogram = Distogram(num_bins=num_bins, min_dist=min_dist, max_dist=max_dist)
        self.pdbdata = PDBFileHandler.fromfilenames(self.file_list_path)
        if self.pdbdata is None:
            raise ValueError(f"Failed to load PDB data from {self.file_list_path}")
    
    def get_distogram(self, index=None, show=True):
        """
        获取指定索引的PDB条目的distogram，并可选地显示结果。
        
        参数:
            index (int, optional): 要处理的PDB条目索引。默认为None，表示遍历所有条目。
            show (bool, optional): 是否显示distogram。默认为True。
            
        返回:
            list: 包含处理后的distogram数据。
        """
        if index is None:
            # 如果index为None，遍历所有PDB条目
            for i in range(len(self.pdbdata.data)):
                self.get_distogram(index=i, show=show)
            return
        
        if index >= len(self.pdbdata.data):
            raise IndexError(f"Index {index} out of range for PDB data")
        
        pdb_entry = self.pdbdata.data[index]
        input_tensor = pdb_entry.atom3
        
        # 生成distogram
        distogram_result = self.distogram.generate_distogram_from_resatom3(input_tensor)
        
        # 应用掩码
        masked_result = []
        for item in distogram_result:
            masked_item = self.mask_processor.mask_distogram(
                item,
                mask_value=0,  # 掩码值
                maskfile=pdb_entry.file_path,
                pr_mask_value=[2, 3]  # 指定掩码值
            )
            masked_item = self.mask_processor.mask_distogram_link(
                masked_item,
                mask_value= 0,
                maskfile=pdb_entry.file_path
            )
            masked_result.append(masked_item)
        
        if show:
            # 绘制distogram
            self.distogram.plot_distogram(masked_result)
        
        return masked_result

def main():
    processor = PDBD('./filelist.txt')
    processor.get_distogram()  # 处理所有PDB条目并显示distogram

if __name__ == "__main__":
    main()