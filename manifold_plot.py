# import numpy as np  #导入numpy包，用于生成数组
# import seaborn as sns  #习惯上简写成sns
# import pandas as pd
# import matplotlib.pyplot as plt
# sns.set()           #切换到seaborn的默认运行配置
from sklearn import manifold, datasets
#
sr_points, sr_color = datasets.make_swiss_roll(n_samples=1000, random_state=0)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)
# ax.scatter(
#     sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8, cmap='jet'
# )
# ax.scatter(8, -0.8860, 12.4128, c=0, s=50, alpha=0.8, cmap='jet')
# ax.set_title("Swiss Roll in Ambient Space")
# ax.view_init(azim=-66, elev=12)
# _ = ax.text2D(0.8, 0.05, s="n_samples=15000", transform=ax.transAxes)
# plt.show()



# -*- coding: utf-8 -*-
# author:           inspurer(月小水长)
# pc_type           lenovo
# create_date:      2019/5/25
# file_name:        3DTest
# github            https://github.com/inspurer
# 微信公众号         月小水长(ID: inspurer)

"""
绘制3d图形
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 定义figure
fig = plt.figure()
# 创建3d图形的两种方式
# 将figure变为3d
ax = Axes3D(fig)

#ax = fig.add_subplot(111, projection='3d')

# 定义x, y
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)

# 生成网格数据
X, Y = np.meshgrid(x, y)

# 计算每个点对的长度
R = np.sqrt(X ** 2 + Y ** 2)
# 计算Z轴的高度
Z = np.sin(R)

# 绘制3D曲面


# rstride:行之间的跨度  cstride:列之间的跨度
# rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现


# cmap是颜色映射表
# from matplotlib import cm
# ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
# cmap = "rainbow" 亦可
# 我的理解的 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
# 你也可以修改 rainbow 为 coolwarm, 验证我的结论
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))

# 绘制从3D曲面到底部的投影,zdir 可选 'z'|'x'|'y'| 分别表示投影到z,x,y平面
# zdir = 'z', offset = -2 表示投影到z = -2上
# ax.contour(X, Y, Z, zdir = 'z', offset = -2, cmap = plt.get_cmap('rainbow'))

# 设置z轴的维度，x,y类似
ax.set_zlim(-2, 2)

plt.show()