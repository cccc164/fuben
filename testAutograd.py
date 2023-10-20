import matplotlib.pyplot as plt
import numpy as np
# 定义字体font1
font1 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 15,
}

# depth = [1, 2, 3]
# AUC_depth = [0.694, 0.769, 0.855]
#
# # plt.figure(figsize=(5, 5), dpi=80)
# plt.bar(depth, AUC_depth, width=0.1, align='center')
# plt.xticks(depth, fontproperties='Times New Roman', size = 12)
# plt.yticks(fontproperties='Times New Roman', size = 12)
# plt.xlabel('layer depth', font1)
# plt.ylabel('AUC', font1)
# plt.show()


# depth = ['32-16-8', '64-32-16', '128-64-32']
# AUC_depth = [0.847, 0.848, 0.855]
#
# # plt.figure(figsize=(5, 5), dpi=80)
# plt.bar(depth, AUC_depth, width=0.1, align='center')
# plt.xticks(depth, fontproperties='Times New Roman', size = 12)
# plt.yticks(fontproperties='Times New Roman', size = 12)
# plt.xlabel('the structure of convolutional kernels', font1)
# plt.ylabel('AUC', font1)
# plt.ylim(0.835, 0.858)
# plt.show()


# depth = [0, 1, 2, 3, 4, 5, 6, 7]
# AUC_depth = [0.694, 0.700, 0.801, 0.833, 0.854, 0.835, 0.834, 0.829]
#
# # plt.figure(figsize=(5, 5), dpi=80)
# plt.plot(depth, AUC_depth, '--r', marker='s', mec='k', mfc='limegreen')
# plt.xticks(depth, fontproperties='Times New Roman', size = 12)
# plt.yticks(fontproperties='Times New Roman', size = 12)
# plt.xlabel(r'$\lambda$', font1)
# plt.ylabel('AUC', font1)
# plt.ylim(0.65, 0.9)
# plt.show()


y_major_ticks_top = np.linspace(0.8, 0.930, 7)
y_minor_ticks_top = np.linspace(0.8, 0.930, 31)
# 绘制卷积窗口-AUC曲线
depth = [i for i in range(2, 11)]
AUC_depth = [0, 0.875, 0, 0.901, 0, 0.851, 0,  0.862, 0]

# plt.figure(figsize=(5, 5), dpi=80)
plt.grid(True, zorder=0, linestyle='--')
plt.bar(depth, AUC_depth, width=0.6, align='center', zorder=10, color="#8faadc", edgecolor='#4876c5')
plt.xticks(depth, fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.xlabel('Kernel size', font1, fontweight='bold', labelpad=7)
plt.ylabel('AUC values', font1, fontweight='bold', labelpad=7)
plt.ylim(0.8, 0.930)
# 保存图形为PNG并指定PPI
plt.savefig('kernel_size_AUC.tif', dpi=300)  # 将图形保存为PNG，分辨率为300 PP
plt.show()



# # 绘制卷积窗口-AUC曲线
# depth = [1, 2, 3]
# AUC_depth = [0.862, 0.875, 0.901]
#
# # plt.figure(figsize=(5, 5), dpi=80)
# plt.grid(True, zorder=0, linestyle='--')
# plt.bar(depth, AUC_depth, width=0.15, align='center', zorder=10, color="#8faadc", edgecolor='#4876c5')
# plt.xticks(depth, fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xlabel('Network depth', font1, fontweight='bold', labelpad=7)
# plt.ylabel('AUC values', font1, fontweight='bold', labelpad=7)
# plt.ylim(0.8, 0.930)
# # 保存图形为PNG并指定PPI
# plt.savefig('Network_depth_AUC.eps', dpi=300)  # 将图形保存为PNG，分辨率为300 PP
# plt.show()


# # 绘制卷积核数量-AUC曲线
# depth = ['16-8-4', '32-16-8', '64-32-16', '128-64-32']
# AUC_depth = [0.857, 0.901, 0.854, 0.821]
#
# # plt.figure(figsize=(5, 5), dpi=80)
# plt.grid(True, zorder=0, linestyle='--')
# plt.bar(depth, AUC_depth, width=0.23, align='center', zorder=10, color="#8faadc", edgecolor='#4876c5')
# plt.xticks(depth, fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.xlabel('the structure of convolutional kernels', font1, fontweight='bold', labelpad=7)
# plt.ylabel('AUC values', font1, fontweight='bold', labelpad=7)
# plt.ylim(0.8, 0.930)
# # 保存图形为PNG并指定PPI
# plt.savefig('structure_AUC.eps', dpi=300)  # 将图形保存为PNG，分辨率为300 PP
# plt.show()

