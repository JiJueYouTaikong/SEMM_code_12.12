import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
od_data = np.load('data/WH_OD_1km_110region.npy')        # 形状为[168, 110, 110]

# 数据的最小值、最大值、均值、标准差
min_val = np.min(od_data)
max_val = np.max(od_data)
mean_val = np.mean(od_data)
std_val = np.std(od_data)
median_val = np.median(od_data)

print(f"最小值: {min_val}")
print(f"最大值: {max_val}")
print(f"均值: {mean_val}")
print(f"标准差: {std_val}")
print(f"中位数: {median_val}")
zeros = np.count_nonzero(od_data == 0)
print(zeros)
nonezeors = np.count_nonzero(od_data != 0)
print(nonezeors)

# 选择4个均匀分布的时间步
num_time_steps = od_data.shape[0]
num_selected_steps = 25

# 使用 np.linspace 生成 4 个均匀分布的时间步
selected_time_steps = np.linspace(0, num_time_steps - 1, num_selected_steps, dtype=int)

# 绘制选择的时间步的热力图
for t in selected_time_steps:
    current_data = od_data[t, :, :]

    plt.figure(figsize=(8, 6))
    sns.heatmap(current_data, cmap='YlGnBu', cbar=True)
    plt.title(f'Heatmap of OD Data at Time Step {t + 1}')
    plt.xlabel('Region Index')
    plt.ylabel('Region Index')
    plt.show()
# 1. 计算时间步上的平均值
# 对 [168, 110, 110] 的数据在第一个维度（时间步）上求均值
average_data = np.mean(od_data, axis=0)

# 2. 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(average_data, cmap='YlGnBu', cbar=True)
plt.title('Average Heatmap of OD Data Over Time Steps')
plt.xlabel('Region Index')
plt.ylabel('Region Index')
plt.show()


# # 计算20%的时间步数量
# num_time_steps = od_data.shape[0]
# num_selected_steps = int(num_time_steps * 0.2)

# 随机选择20%的时间步
# selected_time_steps = np.random.choice(num_time_steps, num_selected_steps, replace=False)

# 对于选中的每个时间步，交换最大的前100个值和最小的100个值
# 对每个时间步进行处理
# 对每个时间步进行处理
# 对每个时间步进行处理
# 设定两套随机种子值

# # 设置随机种子
# np.random.seed(42)  # 在这里设置全局随机种子
#
# # 获取矩阵的维度
# n = od_data.shape[1]
#
# # 创建一个全局有效的位置列表，排除对角线和三对角线
# valid_indices = []
#
# # 遍历矩阵的每个元素，并排除对角线和三对角线位置
# for i in range(n):
#     for j in range(n):
#         if i == j or i == j - 1 or i == j + 1:  # 主对角线和上下邻接的对角线
#             continue
#         valid_indices.append(i * n + j)  # 存储有效位置的线性索引
#
# # 生成100个位置并确保不重叠
# global_noise_indices_1 = np.random.choice(valid_indices, size=200, replace=False)
#
# # 从剩余位置中选择50个位置（不重叠）
# remaining_valid_indices = list(set(valid_indices) - set(global_noise_indices_1))
# global_noise_indices_2 = np.random.choice(remaining_valid_indices, size=50, replace=False)
#
# # 对每个时间步进行处理
# for t in range(od_data.shape[0]):
#     current_data = od_data[t, :, :]
#
#     # 获取当前时间步的最大值
#     max_value = np.max(current_data)
#
#     # 1. 获取展平的数据
#     flat_data = current_data.flatten()
#
#     # 2. 将前100个最大的样本点置为0
#     sorted_indices = np.argsort(flat_data)  # 获取排序后的索引
#     # max_indices = sorted_indices[-100:-50]  # 最大的前100个
#     # flat_data[max_indices] = 0  # 将这些位置的值置为0
#
#     # 3. 给前100个位置添加0.1-0.15比例的噪声
#     noise_factor_1 = np.random.uniform(0.02, 0.08, size=global_noise_indices_1.shape)  # 生成噪声系数（在0.1到0.15之间）
#     noise_1 = max_value * noise_factor_1  # 计算噪声（基于当前时间步的最大值）
#
#     # 将噪声添加到指定的位置
#     flat_data[global_noise_indices_1] += noise_1
#
#     # 4. 给前50个位置添加0.2-0.25比例的噪声
#     noise_factor_2 = np.random.uniform(0.10, 0.2, size=global_noise_indices_2.shape)  # 生成噪声系数（在0.2到0.25之间）
#     noise_2 = max_value * noise_factor_2  # 计算噪声（基于当前时间步的最大值）
#
#     # 将噪声添加到指定的位置
#     flat_data[global_noise_indices_2] += noise_2
#
#     # 5. 恢复数据形状
#     od_data[t, :, :] = flat_data.reshape(current_data.shape)
#

# # 加载数据
# od_data = np.load('data/WH_OD_1km_110region.npy')  # 形状为[168, 110, 110]
#
# # 设置随机种子
# # np.random.seed(42)  # 在这里设置全局随机种子
#
# # 获取矩阵的维度
# n = od_data.shape[1]
#
# # 创建一个全局有效的位置列表，排除对角线和三对角线
# valid_indices = []
#
# # 遍历矩阵的每个元素，并排除对角线和三对角线位置
# for i in range(n):
#     for j in range(n):
#         if i == j or i == j - 1 or i == j + 1:  # 主对角线和上下邻接的对角线
#             continue
#         valid_indices.append((i, j))  # 存储有效位置的索引
#
# # 生成150个位置，并确保它们不重叠
# selected_positions = np.random.choice(len(valid_indices), size=150, replace=False)
# selected_positions = [valid_indices[i] for i in selected_positions]
#
# # 将150个位置划分为两组：100对和50对
# group_1_positions = selected_positions[:120]  # 100对位置
# group_2_positions = selected_positions[120:]  # 50对位置
#
# # # 对每个时间步进行处理
# # for t in range(od_data.shape[0]):
# #     current_data = od_data[t, :, :]
# #
# #     # 获取当前时间步的最大值
# #     max_value = np.max(current_data)
# #
# #     # 1. 获取展平的数据
# #     flat_data = current_data.flatten()
# #
# #     # 2. 给100对位置添加噪声（噪声比例在0.1到0.15之间）
# #     for (i, j) in group_1_positions:
# #         # 对称位置 (i, j) 和 (j, i)
# #         idx_ij = i * n + j
# #         idx_ji = j * n + i
# #
# #         # 为这对位置生成噪声（在0.1到0.15之间）
# #         noise_factor = np.random.uniform(0.01, 0.06)  # 生成噪声系数（在0.1到0.15之间）
# #         noise = max_value * noise_factor  # 计算噪声（基于当前时间步的最大值）
# #         noise_1 =  max_value * np.random.uniform(0.03, 0.05) + noise
# #         # 将噪声添加到指定的位置
# #         flat_data[idx_ij] += noise
# #         flat_data[idx_ji] += noise_1  # 对称位置添加相同的噪声
# #
# #     # 3. 给50对位置添加噪声（噪声比例在0.2到0.25之间）
# #     for (i, j) in group_2_positions:
# #         # 对称位置 (i, j) 和 (j, i)
# #         idx_ij = i * n + j
# #         idx_ji = j * n + i
# #
# #         # 为这对位置生成噪声（在0.2到0.25之间）
# #         noise_factor = np.random.uniform(0.08, 0.25)  # 生成噪声系数（在0.2到0.25之间）
# #         noise = max_value * noise_factor  # 计算噪声（基于当前时间步的最大值）
# #         noise_1 =  max_value * np.random.uniform(0.05, 0.20) + noise
# #         # 将噪声添加到指定的位置
# #         flat_data[idx_ij] += noise
# #         # flat_data[idx_ji] += noise_1    # 对称位置添加相同的噪声
# #
# #     # 4. 恢复数据形状
# #     od_data[t, :, :] = flat_data.reshape(current_data.shape)
#
#
# # 数据的最小值、最大值、均值、标准差
# min_val = np.min(od_data)
# max_val = np.max(od_data)
# mean_val = np.mean(od_data)
# std_val = np.std(od_data)
# median_val = np.median(od_data)
# zeros = np.count_nonzero(od_data == 0)
# print(zeros)
# nonezeors = np.count_nonzero(od_data != 0)
# print(nonezeors)
# print(f"最小值: {min_val}")
# print(f"最大值: {max_val}")
# print(f"均值: {mean_val}")
# print(f"标准差: {std_val}")
# print(f"中位数: {median_val}")
#
#
#
#
# # np.save('data/WH_OD_1km_110region_1.npy',od_data)
#
# # 选择4个均匀分布的时间步
# num_time_steps = od_data.shape[0]
# num_selected_steps = 3
#
# # 使用 np.linspace 生成 4 个均匀分布的时间步
# selected_time_steps = np.linspace(0, num_time_steps - 1, num_selected_steps, dtype=int)
#
# # 绘制选择的时间步的热力图
# # for t in selected_time_steps:
# #     current_data = od_data[t, :, :]
# #
# #     plt.figure(figsize=(8, 6))
# #     sns.heatmap(current_data, cmap='YlGnBu', cbar=True)
# #     plt.title(f'Heatmap of OD Data at Time Step {t + 1}')
# #     plt.xlabel('Region Index')
# #     plt.ylabel('Region Index')
# #     plt.show()
# # # 1. 计算时间步上的平均值
# # # 对 [168, 110, 110] 的数据在第一个维度（时间步）上求均值
# average_data = np.mean(od_data, axis=0)
#
# # 2. 绘制热力图
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(average_data, cmap='YlGnBu', cbar=True)
# # plt.title('Average Heatmap of OD Data Over Time Steps')
# # plt.xlabel('Region Index')
# # plt.ylabel('Region Index')
# # plt.show()