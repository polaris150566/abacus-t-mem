import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class Distogram1d():
    def __init__(self):
        """适合形状为(B,L)的输入,将会对L层的数据进行计算distogram"""
        pass
    
        

class Distogram_coords():
    def __init__(self, input_tensor=None):
        """
        初始化Distogram类，处理输入张量并检查其合法性。
        
        参数：
            input_tensor (torch.Tensor, optional): 输入张量，默认为None。
        """
        self.batch_distogram = []
        self.input = None
        if input_tensor is not None:
            self.input = self.process_input(input_tensor)
            
    def get_input(self,input_tensor):
        self.input = self.process_input(input_tensor)
        return self
    
    def process_input(self, input_tensor):
        """
        处理输入张量，检查其维度、数据类型等是否合法。
        参数：input_tensor (torch.Tensor): 输入三维或四维张量。
        返回：一个四维张量，如果输入时三维的，将其广播成4维的
        """
        # 检查输入是否为张量
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input must be a tensor")
        
        # 检查张量的维度
        if input_tensor.dim() not in [3, 4]:
            raise ValueError("Input should be 3D or 4D tensor")
        
        # 检查数据类型
        if not (torch.is_floating_point(input_tensor) or 
                input_tensor.dtype in [torch.int32, torch.int64]):
            raise TypeError("Input tensor must be int or float type")
        
        # 检查是否存在NaN或Inf
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            raise ValueError("Input tensor contains NaN or Inf values")
        
        # 处理三维或四维张量
        if input_tensor.dim() == 3:#输入的时（L,3,4)
            return input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 4:#输入的是（B,L,3,4)
            return input_tensor
        else:
            raise ValueError("Invalid tensor dimensions")
        return
    
    def generate_distogram_from_resatom3(self):
        """
        从蛋白质的resatom3中生成distogram，直接取用每个res的ca作为残基坐标
        返回的是一个含有多张distogram的数组
        """
        if self.input is None:
            raise ValueError("No input tensor processed yet")
        
        # 提取CA原子坐标（假设CA在第二个位置）
        ca_atoms = self.input[:, :, 1, :]  # 形状为 (B, L, 3)
        
        batch_distogram = []
        for batch in ca_atoms:
            batch_distogram.append(self.generate_distogram(batch))
        
        self.batch_distogram = batch_distogram
        
        return self
    
    def generate_distogram(self, input_tensor):
        """
        生成Distogram矩阵，记录每对残基之间的距离信息。
        参数：input_tensor (torch.Tensor): 形状为 (L, 3) 的张量
        返回：torch.Tensor: 形状为 (L, L) 的Distogram矩阵
        """
        if input_tensor.dim() != 2 or input_tensor.size(1) != 3:
            raise ValueError("Input tensor must have shape (L, 3)")
        
        num_residues = input_tensor.size(0)
        
        # 使用广播计算所有残基对之间的距离（高效实现）
        diff = input_tensor.unsqueeze(1) - input_tensor.unsqueeze(0)
        distogram = torch.norm(diff, dim=-1)
        
        # 对角线设为0
        distogram.diagonal().zero_()
        
        return distogram
    
    def render(self, distogram = None, title='Distogram'):
        """
        绘制Distogram热力图。
        
        参数：
            distogram (torch.Tensor): 形状为 (L, L) 的Distogram矩阵数组
            title (str): 图像标题
        """
        if not distogram:
            distogram = self.batch_distogram
            #print(distogram)
        for item in distogram:
            if item.dim() != 2:
                raise ValueError("Distogram must be 2D matrix")
            
            plt.figure(figsize=(8, 6))
            plt.imshow(item, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Distance (Å)')
            plt.title(title)
            plt.xlabel('Residue Index')
            plt.ylabel('Residue Index')
            plt.show()
        # return self
        
    def result(self):
        return self.batch_distogram

if __name__ == '__main__':
    # 创建一个模拟的输入张量（四维）
    input_tensor = torch.randn(2, 10, 3, 3)  # 2个蛋白质，每个有10个残基，每个残基有3个原子，每个原子有3个坐标值
    
    # 创建Distogram对象
    distogram = Distogram_coords()
    
    distogram.get_input(input_tensor).generate_distogram_from_resatom3().render()