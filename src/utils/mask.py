import numpy as np
import torch
import torch.nn as nn
import os
import torch

class Mask_Processor():
    def __init__(self,filepath = './213mask.npy') -> None:
        self.mask_data = (np.load(file=filepath,allow_pickle=True)).item()
        self.pr_mask = None
    
    def npy_loader(self, pr_name):
        if pr_name == '':
            raise ValueError("pr_name should not be empty")
        
        if pr_name not in self.mask_data:
            raise KeyError(f"mask '{pr_name}' not in data")
        self.pr_mask = self.mask_data[pr_name]
        
    def mask_1d(self, input_tensor, maskfile='',mask_value = 0,pr_mask_value = [2,3]):
        """
        输入的第一个参数是被掩码的张量，形状是 (L, ...)，其中 L 是第一维度的长度
        maskfile 是掩码对应的索引文件名,如果找不到文件的默认为全1张量
        返回值是被掩码的张量
        
        目前只接受一维的掩码
        
        """
        with torch.no_grad():
            def process_filename(filename):
                basename = os.path.basename(filename)
                name, ext = os.path.splitext(basename)
                if ext.lower() != '.pdb':
                    raise ValueError(f"文件后缀名不是 .pdb，而是 {ext}")
                return name

            # 处理掩码文件名
            if maskfile:
                print(maskfile)
                maskfile = process_filename(maskfile)
                
                self.npy_loader(maskfile)

                pr_mask = torch.tensor(self.pr_mask).long()
            else:
                # 如果没有提供掩码文件，假设所有位置都有效
                pr_mask = torch.ones(input_tensor.size(0), dtype=torch.long)

            # 创建掩码张量
            mask = ~torch.isin(pr_mask , torch.tensor(pr_mask_value))
            mask_modified = torch.where(mask, mask_value, 1)

            if mask_modified.size(0) != input_tensor.size(0):
                raise ValueError(" inconsistent with the first dimension length of the tensor")

            mask_broadcasted = mask_modified.view(-1, *([1] * (input_tensor.dim() - 1)))
            mask_broadcasted = mask_broadcasted.expand_as(input_tensor)

            result = input_tensor * mask_broadcasted

            return result

    def mask_distogram(self, input_distogram, maskfile='',mask_value = 0,pr_mask_value =[2,3]):
        """给distogram打掩码，
        输入：inputdistogram：输入的distogram张量，形状为(L,L)
        maskfile：输入的掩码文件，通常从data【0】。filename中获得
        maskvalue：被掩去的值，通常是0
        prmaskvalue:想要掩码的部分，1是抗原，2，3分别是两条链的CDR，0是重链或者轻链的frame"""
        def process_filename(filename):
                basename = os.path.basename(filename)
                name, ext = os.path.splitext(basename)
                if ext.lower() != '.pdb':
                    raise ValueError(f"文件后缀名不是 .pdb，而是 {ext}")
                return name

            # 处理掩码文件名
        if maskfile:
            print(maskfile)
            maskfile = process_filename(maskfile)
            
            self.npy_loader(maskfile)

            pr_mask = torch.tensor(self.pr_mask).long()
        else:
            # 如果没有提供掩码文件，假设所有位置都有效
            pr_mask = torch.ones(input_distogram.size(0), dtype=torch.long)

        # 创建掩码张量
        mask = ~torch.isin(pr_mask , torch.tensor(pr_mask_value))
        outer_product = torch.outer(mask, mask)
        mask_modified = torch.where(outer_product == 1, 1, mask_value)

        return input_distogram * mask_modified
    def mask_distogram_link(self,distogram_tensor, maskfile='',mask_value = 0):
        """
        对输入的distogram创建掩码。
        
        参数:
            input_tensor (torch.Tensor): 一维张量，形状为 (L,)，里面有每个氨基酸的状态，1代表抗原，其余数字代表抗体上的残基。
            distogram_tensor (torch.Tensor): 二维张量，形状为 (L, L)，输入的另一个二维张量。
            mask_value (int, optional): 需要设置的掩码值，默认为0。
        
        返回:
            torch.Tensor: 处理后的二维张量，形状为 (L, L)。
        """
        def process_filename(filename):
                basename = os.path.basename(filename)
                name, ext = os.path.splitext(basename)
                if ext.lower() != '.pdb':
                    raise ValueError(f"文件后缀名不是 .pdb，而是 {ext}")
                return name
        if maskfile:
            print(maskfile)
            maskfile = process_filename(maskfile)
            
            self.npy_loader(maskfile)

            pr_mask = torch.tensor(self.pr_mask).long()
        else:
            pr_mask = torch.ones(distogram_tensor.size(0), dtype=torch.long)
        
        
        outer_product = pr_mask.unsqueeze(1) * pr_mask.unsqueeze(0)
        
        # 创建掩码矩阵，初始化为1
        mask = torch.ones_like(outer_product)
        
        # 找到一边是抗原（1），另一边是抗体（非1）的元素位置
        antigen_positions = torch.where(pr_mask == 1, 1, -1)
        
        mask = (antigen_positions.unsqueeze(1)*antigen_positions.unsqueeze(0)-mask)/2*(1-mask_value)+mask
        
        # 将掩码与 distogram_tensor 逐元素相乘
        masked_distogram = distogram_tensor * mask
        
        return masked_distogram 
        
    def log_mast_status(self):
        print(self.pr_mask)
        pass
    

    
def main():
    mask = Mask_Processor()
    mask.npy_loader('1adq_H_L_A')
    mask.log_mast_status()
if __name__ == "__main__":
    main()