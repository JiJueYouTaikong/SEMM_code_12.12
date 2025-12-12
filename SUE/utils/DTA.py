import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.distributions import Gumbel
# 参数设置
capacity = 500  # 路段容量
alpha = 0.15  # BPR函数的alpha参数
beta = 4  # BPR函数的beta参数
lambda_param = 1  # Logit模型灵敏度参数
max_iter = 1  # 最大迭代次数
tol = 1e-3  # 收敛阈值

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
        print(f"logit模型：{origin}")

        for destination in range(N):
            if od_matrix_t[origin, destination] > 0:
                # print('--------------------------------')
                # print(f"{origin} -> {destination}存在OD流量")
                demand = od_matrix_t[origin, destination]
                # print("开始寻找路径")
                # 计算从起点到所有目的地的路径通行时间
                paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=1))
                if not paths:  # 无路径跳过
                    # print("--无路径--")
                    continue
                # print(f"--已搜索到{len(paths)}条路径集合--")

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

                # print('已按logit模型分配到路段')

    # print("一个时间步的OD流量已分配完成")

    # 更新路段的通行时间
    for u, v in G.edges:
        flow = G[u][v]['flow']
        t0 = dist_matrix[u, v]
        G[u][v]['distance'] = bpr_function(t0, flow, capacity)

    # print("OD流量分配完成后根据流量和BPR更新时间")

    return new_flows



def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol):
    '''
    :param od_matrix:   N, N
    :param adj_matrix:
    :param dist_matrix:
    :param lambda_param:
    :param max_iter:
    :param tol:
    :return: 分配流量 N,N
    '''
    N, _ = od_matrix.shape

    flow_results = np.zeros((N, N))  # 存储流量矩阵

    G = initialize_graph(adj_matrix, dist_matrix) # 重新初始化有向图G



    for iteration in range(max_iter): # for loop
        # 分配流量
        # print("进入迭代分配")
        flow_matrix = logit_traffic_assignment(G, od_matrix, lambda_param)
        # print("完成迭代分配")

    flow_results = flow_matrix


    return flow_results