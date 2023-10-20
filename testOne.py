import numpy as np

# 创建一个示例的二维矩阵
matrix = np.array([[5, 2, 8],
                  [3, 7, 1],
                  [9, 4, 6]])

# 设置你想要的前几个百分比，这里设置为前30%
top_percentage = 0.3

# 计算要保留的索引数
num_elements = matrix.size
num_to_keep = int(top_percentage * num_elements)

# 使用argsort对矩阵中的值进行排序，并返回索引
sorted_indices = np.argsort(matrix, axis=None)

# 创建一个全零的与原矩阵相同形状的矩阵
result_matrix = np.zeros_like(matrix)

# 将前num_to_keep个位置设置为1
result_matrix.flat[sorted_indices[:num_to_keep]] = 1

# 打印结果矩阵
print("结果矩阵:")
print(result_matrix)
