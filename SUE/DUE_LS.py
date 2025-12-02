import time

import numpy as np
import networkx as nx
import torch
from matplotlib import pyplot as plt

def convert_matrix(flow, adj_matrix):
    # 找到 adj_matrix 中非零元素的索引
    non_zero_indices = np.nonzero(adj_matrix)
    # 提取 flow 中对应位置的值
    modified_flow = flow[non_zero_indices]
    # 将结果转换为 [L, 1] 的形状
    modified_flow = modified_flow.reshape(-1, 1)
    return modified_flow
    # A = [m,m]时
    # n = adj_matrix.shape[0]
    # flow = flow.reshape(n * n, -1)
    # return flow





# 计算 RMSE、MAE 和 MAPE
import numpy as np


def calculate_rmse_mae(predictions, targets):
    # 计算均方误差
    mse = np.mean((predictions - targets) ** 2)
    # 计算均方根误差
    rmse = np.sqrt(mse)
    # 计算平均绝对误差
    mae = np.mean(np.abs(predictions - targets))

    # 计算 MAPE，避免除以零
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(
            np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = 0.0

    return rmse, mae, mape


def least_squares_solve(f, A, N):
    # 计算 A^T
    # print("A",A.shape)
    # print("f",f.shape)

    A_T = np.transpose(A)
    # print("A_T",A_T.shape)

    # 计算 A^T * A
    A_T_A = np.dot(A_T, A)
    # print("A_T_A",A_T_A.shape)

    # 计算 A^T * A 的逆
    pinv_A_T_A = np.linalg.pinv(A_T_A)
    # print("pinv_A_T_A(-1)",pinv_A_T_A.shape)

    # 计算 (A^T * A)^(-1) * A^T
    pinv_A_T_A_A_T = np.dot(pinv_A_T_A, A_T)
    # print("pinv_A_T_A(-1)_A_T",pinv_A_T_A_A_T.shape)

    # 计算 t
    t = np.dot(pinv_A_T_A_A_T, f)

    t = t.reshape(N,N)
    return t

def get_allocation_matrix(od_matrix, adj_matrix, dist_matrix):
    # n = od_matrix.shape[0]
    # nn = n * n
    # # 初始化分配矩阵 A
    # A = np.zeros((nn, nn))
    #
    # # 创建有向图
    # G = nx.DiGraph()
    # for i in range(n):
    #     for j in range(n):
    #         if adj_matrix[i, j] != 0:
    #             G.add_edge(i, j, weight=dist_matrix[i, j])
    #
    # # 遍历所有 OD 对
    # for origin in range(n):
    #     for destination in range(n):
    #         col_index = origin * n + destination
    #         try:
    #             # 查找最短路径
    #             path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
    #             for k in range(len(path) - 1):
    #                 u, v = path[k], path[k + 1]
    #                 row_index = u * n + v
    #                 A[row_index, col_index] = 1
    #         except nx.NetworkXNoPath:
    #             continue
    #
    # return A
    n = od_matrix.shape[0]
    # 找出邻接矩阵中非零元素的位置
    non_zero_indices = np.nonzero(adj_matrix)
    l = len(non_zero_indices[0])
    # 初始化分配矩阵 A
    A = np.zeros((l, n * n))

    # 创建有向图
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    # 遍历所有 OD 对
    for origin in range(n):
        for destination in range(n):
            col_index = origin * n + destination
            try:
                # 查找最短路径
                path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # 找到该路段在路段列表中的索引
                    road_index = np.where((non_zero_indices[0] == u) & (non_zero_indices[1] == v))[0][0]
                    A[road_index, col_index] = 1
            except nx.NetworkXNoPath:
                continue

    return A


# 示例数据
N = 110
num_samples = 35

od_matrix = np.load("data/OD_完整批处理_3.17_Final.npy")  # [T,N,N]
od_matrix = od_matrix[-num_samples:].astype(np.float32)

flow = np.load("data/Link_flow_TNN_MSA-SUE_logit3_6_14.npy")  # [T,N,N]
flow = flow[-num_samples:].astype(np.float32)


adj_matrix = np.load("data/adj110_3.17.npy")

dist_matrix = np.load("data/dist110_3.17.npy")


rmse_total = 0
mae_total = 0
mape_total = 0


start_time = time.time()

# 初始化列表收集每个时间步预测结果
t_pred_list = []

A = get_allocation_matrix(od_matrix[0], adj_matrix, dist_matrix)

for i in range(num_samples):
    print(f"----------- 样本{i}------------")
    # 求解分配矩阵 A

    print("f = A*t , 利用最小二乘法求解t_pred")
    print(A.shape)

    flow_i = convert_matrix(flow[i], adj_matrix)

    t = least_squares_solve(flow_i, A, N)

    t_pred_list.append(t)

    rmse, mae, mape = calculate_rmse_mae(t, od_matrix[i])
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}")
    rmse_total += rmse
    mae_total += mae
    mape_total += mape

# 拼接为 [num_samples, N, N] 的 numpy 数组并保存
t_pred_array = np.stack(t_pred_list, axis=0).astype(np.float32)
np.save("../可视化/测试集TNN/Pred-DUE-LS.npy", t_pred_array)

# 计算平均评估指标
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples

print(f"Total Test RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}")

end_time = time.time()
time_difference_seconds = end_time - start_time
times = time_difference_seconds / 60
print(f"耗时{times:.2f}min")

log_filename = f"log/DUE_最小二乘.log"
with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Samples: {num_samples}, Total RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}, Times: {times}\n")


