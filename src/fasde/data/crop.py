import torch
import numpy as np
from scipy.spatial import KDTree
import  traceback
import time

class Ellipsoid_crop:
    def __init__():
        pass

    @classmethod
    def transform_points(cls,points, affine_matrix = np.eye(3)):
        """对输入的点集做指定的线性变换

        Args:
            points (_type_): _description_
            affine_matrix (_type_, optional): _description_. Defaults to np.eye(3).

        Returns:
            _type_: _description_
        """
        # 将输入转换为 NumPy 数组以进行矩阵运算
        points_np = points.numpy()

        # 应用仿射变换
        transformed_points = np.dot(points_np, affine_matrix.T)

        # 将结果转换回 PyTorch 张量
        transformed_points_tensor = torch.from_numpy(transformed_points[:, :3])

        return transformed_points_tensor

    @classmethod
    def find_k_nearest_neighbors(cls, points, query_point, k):
        """crop的主函数，其中points应是已经仿射变换好的点集，

        Args:
            points (_type_):目标点集
            query_point (_type_): 中心点
            k (_type_): 接受范围，k个以内的将会被接受

        Returns:
            _type_: 返回一个序列，表示被接受的点再原来点集内的索引
        """
        points = np.array(points)
        query_point = np.array(query_point)
        tree = KDTree(points)
        distances, indices = tree.query(query_point, k)
        return indices

    @classmethod
    def generate_center_point(cls, points, std_dev = 2):
        """根据输入的点集生成一个中心点

        Args:
            points (可以是普通数组): 输入的点集
            std_dev (float, optional): _description_.

        Returns:
            _type_: 一个np数组
        """
        np.random.seed(int(time.time()))
        points = points.numpy()
        center = np.mean(points, axis=0)

        # 添加高斯噪声
        noisy_center = center + np.random.normal(loc=0, scale=std_dev, size=center.shape)

        return noisy_center

    @classmethod
    def crop_and_search_return_idx(cls, points, affine_matrix, k = 200, std_dev = 20):
        """_summary_

        Args:
            points (_type_): 输入的点集，可以是普通数组
            k ( int ):选择最近的k个近邻
            affine_matrix:变换矩阵

        Returns:
            _type_: 返回一个一维张量，内容是points中被接受的点的索引
        """
        try:
            transformed_points = cls.transform_points(points, affine_matrix = affine_matrix)

            center_point = cls.generate_center_point(transformed_points,std_dev = std_dev)
            # print(transformed_points, center_point)

            indices = cls.find_k_nearest_neighbors(transformed_points, center_point, k)

            indices_tensor = torch.from_numpy(indices)
            return indices_tensor

        except Exception as e:
            traceback.print_exc(e)
            import pdb; pdb.set_trace()

    @classmethod
    def generate_ellipsoid_affine_matrix(cls, long_axis_vector, eccentricity):
        """对长轴向量（实际上就是膜的法向量）进行归一化，
        之后假设短轴长度是1，并根据离心率算出相应的长轴长度
        之后构建对角矩阵

        Args:
            long_axis_vector (_type_): _description_
            eccentricity (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 归一化长轴向量
        long_axis_vector = long_axis_vector / np.linalg.norm(long_axis_vector)

        # 计算长轴和短轴的比例
        short_axis_length = 1.0
        long_axis_length = short_axis_length / np.sqrt(1 - eccentricity**2)
        tmp_v = long_axis_vector * long_axis_length

        # 构建对角阵,旋转矩阵
        diagonal_matrix = np.diag([ short_axis_length, short_axis_length, short_axis_length/long_axis_length])
        rotation_matrix = cls.b_rodrigues_matrix(long_axis_vector, [0,0,1])
        # print(np.linalg.det(rotation_matrix))


        # 生成仿射变换矩阵
        affine_matrix = np.dot(diagonal_matrix, rotation_matrix)#先旋转后放缩,因为之后要转置，所以直接反过来
        print(np.dot(tmp_v,affine_matrix.T))

        return affine_matrix

    @classmethod
    def b_rodrigues_matrix(cls,u, v):
        """使用罗德里格斯公式计算旋转矩阵，将向量 u 旋转到向量 v 的方向。
        input:
            u -- 原始向量（3D）
            v -- 目标向量（3D）
        return:
            旋转矩阵（3x3 numpy 数组）
        """
        # 归一化向量
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)

        k = np.cross(u, v)
        k = k / np.linalg.norm(k)  # 归一化
        theta = np.arccos(np.dot(u, v))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        kx, ky, kz = k

        rotation_matrix = np.array([
            [cos_theta + kx**2 * (1 - cos_theta)     , kx*ky*(1 - cos_theta) - kz*sin_theta    , kx*kz*(1 - cos_theta) + ky*sin_theta ],
            [ky*kx*(1 - cos_theta) + kz*sin_theta    , cos_theta + ky**2 * (1 - cos_theta)     , ky*kz*(1 - cos_theta) - kx*sin_theta ],
            [kz*kx*(1 - cos_theta) - ky*sin_theta    , kz*ky*(1 - cos_theta) + kx*sin_theta    , cos_theta + kz**2 * (1 - cos_theta)  ]
        ])

        return rotation_matrix

# 示例用法
if __name__ == "__main__":
    # 定义椭球参数
    long_axis_vector = np.array([35,43, 8])  # 长轴方向向量
    eccentricity = 0.5                      # 离心率

    # 生成仿射变换矩阵
    affine_matrix = Ellipsoid_crop.generate_ellipsoid_affine_matrix(long_axis_vector, eccentricity)
    # print(affine_matrix)
    # print(np.linalg.det(affine_matrix))

    # 创建散点集
    points = torch.tensor([[21.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0],
                          [10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0]])

    # 执行 crop_and_search_return_idx 函数
    k = 3
    indices = Ellipsoid_crop.crop_and_search_return_idx(points, affine_matrix, k, std_dev=10)
    print(f"最近的 {k} 个散点的索引:", indices)