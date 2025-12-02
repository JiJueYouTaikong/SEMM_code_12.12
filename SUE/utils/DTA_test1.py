import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

# 参数设置
capacity = 500  # 路段容量
alpha = 0.15  # BPR函数的alpha参数
beta = 4  # BPR函数的beta参数
lambda_param = 1  # Logit模型灵敏度参数
max_iter = 1  # 最大迭代次数

adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]

# 初始化图
def initialize_graph(adj_matrix, dist_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for i, j in G.edges:
        G[i][j]['distance'] = dist_matrix[i, j]
        G[i][j]['flow'] = 0  # 初始流量为0
    return G

# BPR函数计算通行时间
def bpr_function(t0, v, c):
    return t0 * (1 + alpha * (v / c) ** beta)

# Logit模型分配流量
def logit_traffic_assignment(G, od_matrix_t, lambda_param):
    N = od_matrix_t.shape[0]
    new_flows = np.zeros((N, N))  # 存储当前时间步的流量矩阵

    for origin in range(N):
        print(origin)
        for destination in range(N):

            if od_matrix_t[origin, destination] > 0:
                demand = od_matrix_t[origin, destination]
                paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=2))
                if not paths:  # 无路径跳过
                    continue

                path_times = []
                for path in paths:
                    time = sum(G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                    path_times.append(time)

                # 计算Logit模型选择概率
                path_probs = np.exp(-lambda_param * np.array(path_times))
                path_probs /= path_probs.sum()  # 归一化

                # 将流量按概率分配到路径
                for path, prob in zip(paths, path_probs):
                    assigned_flow = demand * prob
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        G[u][v]['flow'] += assigned_flow
                        new_flows[u, v] += assigned_flow

    # 更新路段的通行时间
    for u, v in G.edges:
        flow = G[u][v]['flow']
        t0 = dist_matrix[u, v]
        G[u][v]['distance'] = bpr_function(t0, flow, capacity)

    return new_flows

def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter):
    '''
    :param od_matrix:   N, N
    :param adj_matrix:
    :param dist_matrix:
    :param lambda_param:
    :param max_iter:
    :param tol:
    :return: 分配流量 N,N
    '''
    N = int(np.sqrt(od_matrix.size))  # 计算N的值
    od_matrix = od_matrix.reshape((N, N))  # 确保od_matrix是二维矩阵
    flow_results = np.zeros((N, N))  # 存储流量矩阵
    G = initialize_graph(adj_matrix, dist_matrix) # 重新初始化有向图G

    for iteration in range(max_iter): # for loop
        flow_matrix = logit_traffic_assignment(G, od_matrix, lambda_param)
    flow_results = flow_matrix
    return flow_results

# 计算损失函数
def compute_loss(od_matrix_flat, observed_flow, adj_matrix, dist_matrix, lambda_param):
    start_time = time.time()
    N = int(np.sqrt(od_matrix_flat.size))
    od_matrix = od_matrix_flat.reshape((N, N))
    print(f"OD Matrix (min, max): {od_matrix.min()}, {od_matrix.max()}")  # 打印OD矩阵的范围
    predicted_flow = run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter=1)
    loss = np.sum((predicted_flow - observed_flow) ** 2)
    print(f"Loss: {loss}, Time: {time.time() - start_time:.2f}s")
    return loss

def callback(xk):
    print(f"Current parameters (flattened): {xk}")

def optimize_od(initial_od, observed_flow, adj_matrix, dist_matrix, lambda_param, max_iter):
    bounds = [(0, None) for _ in range(initial_od.size)]
    result = minimize(compute_loss, initial_od.flatten(), args=(observed_flow, adj_matrix, dist_matrix, lambda_param),
                      method='L-BFGS-B', options={'maxiter': max_iter}, callback=callback)
    optimized_od = result.x.reshape(initial_od.shape)
    return optimized_od

# 计算RMSE和MAE
def compute_metrics(optimized_od, observed_od):
    rmse = np.sqrt(np.mean((optimized_od - observed_od) ** 2))
    mae = np.mean(np.abs(optimized_od - observed_od))
    return rmse, mae

# 主函数
def main():
    N = adj_matrix.shape[0]
    observed_od = np.random.rand(N, N)  # 实际观测的OD矩阵
    observed_flow = np.random.rand(N, N)  # 实际观测的链路Flow

    # 随机初始化OD矩阵
    initial_od = np.random.rand(N, N)
    print("init od",initial_od)

    # 优化OD矩阵
    optimized_od = optimize_od(initial_od, observed_flow, adj_matrix, dist_matrix, lambda_param, max_iter)

    print("final pred od",optimized_od)
    print("final obs od",observed_od)

    # 计算最终的RMSE和MAE
    rmse, mae = compute_metrics(optimized_od, observed_od)
    print(f"RMSE: {rmse}, MAE: {mae}")

if __name__ == "__main__":
    main()