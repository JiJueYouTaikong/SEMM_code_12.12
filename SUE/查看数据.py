# import numpy as np
#
# dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]
# print(dist_matrix.shape)
# print(dist_matrix)
#
# adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
# print(adj_matrix.shape)
# print(adj_matrix)
#
# # 把邻接矩阵为 0 的元素在距离矩阵中对应位置的元素设为无穷大
# dist_matrix[adj_matrix == 0] = np.inf
#
# print("处理后的距离矩阵:")
# print(dist_matrix)

import numpy as np

# od = np.load("../data/OD_完整批处理_3.17_Final.npy")
#
# avg_od = np.mean(od, axis=0)
#
# np.save("../data/初始OD估计_AVG_NN_5.25.npy", avg_od)

import numpy as np

# 加载数据
flow = np.load('data/Link_flow_TNN_3.23_可微.npy')  # shape: [T, N, N]

# 统计添加前的零值占比与总和
zero_ratio_before = np.sum(flow == 0) / flow.size
sum_before = np.sum(flow)
print(f"原始数据中零值占比: {zero_ratio_before:.4%}")
print(f"原始数据总和: {sum_before:.4f}")

# 噪声参数
noise_level = 0.05  # 5% 噪声强度
std_dev = flow.std() * noise_level
mean = 0.0

# 创建噪声，仅在 flow > 5 的位置添加
noise = np.zeros_like(flow)
mask = flow > 2
noise[mask] = np.random.normal(loc=mean, scale=std_dev, size=np.sum(mask))

# 添加噪声并 clip 非负
flow_noisy = flow + noise
flow_noisy = np.clip(flow_noisy, 0, None)

# 统计添加后的零值占比与总和
zero_ratio_after = np.sum(flow_noisy == 0) / flow_noisy.size
sum_after = np.sum(flow_noisy)
print(f"添加高斯噪声后零值占比: {zero_ratio_after:.4%}")
print(f"添加高斯噪声后总和: {sum_after:.4f}")


