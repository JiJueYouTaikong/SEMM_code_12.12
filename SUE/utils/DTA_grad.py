import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 参数设置
capacity = 500  # 路段容量
alpha = 0.15  # BPR函数的alpha参数
beta = 4  # BPR函数的beta参数
lambda_param = 1  # Logit模型灵敏度参数
max_iter = 1  # 最大迭代次数
tol = 1e-3  # 收敛阈值

adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]

# 将numpy数组转换为torch张量
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, requires_grad=False)
dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, requires_grad=False)

# BPR函数计算通行时间
def bpr_function(t0, v, c):
    return t0 * (1 + alpha * (v / c) ** beta)

# 可微的Logit模型分配流量
def logit_traffic_assignment(adj_matrix, dist_matrix, od_matrix_t, lambda_param):
    N = od_matrix_t.shape[0]
    new_flows = torch.zeros((N, N), dtype=torch.float32, requires_grad=True)

    for origin in range(N):
        for destination in range(N):
            if od_matrix_t[origin, destination] > 0:
                demand = od_matrix_t[origin, destination]
                # 这里需要实现可微的路径搜索，假设我们使用暴力搜索（仅为示例）
                paths = []
                for i in range(2 ** N):
                    path = []
                    node = origin
                    for j in range(N):
                        if (i >> j) & 1:
                            neighbors = torch.nonzero(adj_matrix[node], as_tuple=False).squeeze()
                            if neighbors.numel() > 0:
                                node = neighbors[0].item()
                                path.append(node)
                                if node == destination:
                                    paths.append(path)
                                    break
                if not paths:
                    continue

                path_times = []
                for path in paths:
                    time = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
                    path_times.append(time)

                path_times = torch.tensor(path_times, dtype=torch.float32)
                path_probs = torch.exp(-lambda_param * path_times)
                path_probs /= path_probs.sum()

                for path, prob in zip(paths, path_probs):
                    assigned_flow = demand * prob
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        new_flows[u, v] += assigned_flow

    # 更新路段的通行时间
    for u in range(N):
        for v in range(N):
            if adj_matrix[u, v] > 0:
                flow = new_flows[u, v]
                t0 = dist_matrix[u, v]
                dist_matrix[u, v] = bpr_function(t0, flow, capacity)

    return new_flows

def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol):
    N, _ = od_matrix.shape
    flow_results = torch.zeros((N, N), dtype=torch.float32, requires_grad=True)

    for iteration in range(max_iter):
        flow_matrix = logit_traffic_assignment(adj_matrix, dist_matrix, od_matrix, lambda_param)
        flow_results = flow_matrix

    return flow_results