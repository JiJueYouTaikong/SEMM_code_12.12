import numpy as np
import networkx as nx

def convert_matrix(flow, adj_matrix):
    # 找到 adj_matrix 中非零元素的索引
    non_zero_indices = np.nonzero(adj_matrix)
    # 提取 flow 中对应位置的值
    modified_flow = flow[non_zero_indices]
    # 将结果转换为 [L, 1] 的形状
    modified_flow = modified_flow.reshape(-1, 1)
    return modified_flow

def softmax(x):
    print("softmax输入",x)
    """计算 softmax 概率"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_allocation_matrix(od_matrix, adj_matrix, dist_matrix, cutoff):
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
            print("------------------")
            col_index = origin * n + destination
            paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=cutoff))
            path_costs = []
            for path in paths:
                cost = 0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    cost += dist_matrix[u, v]
                path_costs.append(-cost)  # 取负是因为成本越低越优

            if paths:
                path_probs = softmax(np.array(path_costs))
                for path_idx, path in enumerate(paths):
                    print(path_probs[path_idx])
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # 找到该路段在路段列表中的索引
                        road_index = np.where((non_zero_indices[0] == u) & (non_zero_indices[1] == v))[0][0]
                        A[road_index, col_index] += path_probs[path_idx]

    return A


# 示例数据
N = 10
od_matrix = np.random.rand(N, N)
print(od_matrix)
adj_matrix = np.random.rand(N, N)
dist_matrix = np.random.rand(N, N)
# 将非联通节点距离设为无穷大
dist_matrix[adj_matrix == 0] = np.inf
print(dist_matrix)


cutoff = 2  # 设置简单路径搜索的截断值
# 求解分配矩阵 A
A = get_allocation_matrix(od_matrix, adj_matrix, dist_matrix, cutoff)
print("分配矩阵 A:")
print(A)


f = np.array([
    [0, 10, 10],    [10, 0, 0],   [10, 0, 0]
])
def least_squares_solve(f, A):
    # # 计算最小二乘解
    # # 根据最小二乘法原理，t = (A^T * A)^(-1) * A^T * f
    # A_T = np.transpose(A)
    # A_T_A = np.dot(A_T, A)
    # inv_A_T_A = np.linalg.inv(A_T_A)
    # inv_A_T_A_A_T = np.dot(inv_A_T_A, A_T)
    # t = np.dot(inv_A_T_A_A_T, f)
    # return t
    # 计算 A^T
    print("A",A.shape)
    print("f",f.shape)

    A_T = np.transpose(A)
    print("A_T",A_T.shape)

    # 计算 A^T * A
    A_T_A = np.dot(A_T, A)
    print("A_T_A",A_T_A.shape)

    # 计算 A^T * A 的伪逆
    pinv_A_T_A = np.linalg.pinv(A_T_A)
    print("pinv_A_T_A(-1)",pinv_A_T_A.shape)

    # 计算 (A^T * A)^(-1) * A^T
    pinv_A_T_A_A_T = np.dot(pinv_A_T_A, A_T)
    print("pinv_A_T_A(-1)_A_T",pinv_A_T_A_A_T.shape)

    # 计算 t
    t = np.dot(pinv_A_T_A_A_T, f)
    return t


# f = convert_matrix(f,adj_matrix)
#
# t = least_squares_solve(f, A)
# print("求解的OD",t)
# t = t.reshape(N, N)
# print("OD",t)
# import numpy as np
# from scipy.linalg import inv
# from scipy.sparse.csgraph import dijkstra
#
# # 假设数据
# N = 3  # 区域数量
#
#
# # OD矩阵 (N x N)
# OD_matrix = np.random.randint(50, 200, size=(N, N))
# print(OD_matrix)
# # 邻接矩阵 (N x N)，非零值表示链路存在
# adjacency_matrix = np.random.choice([0, 1], size=(N, N), p=[0.4, 0.6])
# adjacency_matrix = np.array(
# [[0, 1 ,1],
#  [1 ,0 ,0],
#  [1 ,0 ,0]]
# )
# print(adjacency_matrix)
# # 距离矩阵 (N x N)
# distance_matrix = np.random.randint(1, 10, size=(N, N)) * adjacency_matrix
# print(distance_matrix)
#
# # 真实链路流量 (N x N)
# true_link_flows = np.random.randint(50, 200, size=(N, N)) * adjacency_matrix
#
# # 将OD矩阵reshape为[n*n, 1]的向量
# OD_vector = OD_matrix.reshape(-1)
#
# # 将真实链路流量通过邻接矩阵修改为[L, 1]的向量
# link_flows_vector = true_link_flows[adjacency_matrix > 0].reshape(-1)
# L = link_flows_vector.shape[0]
#
# # 步骤1: 计算方差-协方差矩阵 V 和 W
# # 假设OD矩阵的估计值方差与真实值成正比 9
# V = np.diag(OD_vector * 0.1)  # 假设方差为真实值的10%
#
# # 假设链路流量的方差与真实流量成正比
# W = np.diag(link_flows_vector * 0.1)  # 假设方差为真实流量的10%
#
# # 步骤2: 基于最短路径算法生成分配矩阵 A
# # 使用Dijkstra算法计算最短路径
# shortest_path_matrix = dijkstra(distance_matrix, directed=False, return_predecessors=True)
#
# # 初始化分配矩阵 A (L x N^2)
# A = np.zeros((L, N**2))
#
# # 填充分配矩阵 A
# link_index = 0
# for i in range(N):
#     for j in range(N):
#         if i != j:  # 跳过自身到自身的OD对
#             # 找到从i到j的最短路径
#             path = []
#             current = j
#             while current != i:
#                 path.append(current)
#                 current = shortest_path_matrix[1][i, current]
#             path.append(i)
#             path.reverse()
#
#             # 将路径上的链路标记为1
#             for k in range(len(path) - 1):
#                 start = path[k]
#                 end = path[k + 1]
#                 if adjacency_matrix[start, end] > 0:
#                     link_index = \
#                     np.where((adjacency_matrix > 0) & (np.arange(N) == start)[:, None] & (np.arange(N) == end))[0][0]
#                     A[link_index, i * N + j] = 1
#
# print(A)
#
# # 步骤3: 构建广义最小二乘估计器
# # 计算V的逆矩阵
# V_inv = inv(V)
#
# # 计算W的逆矩阵
# W_inv = inv(W)
#
# # 计算A^T * W^-1 * A
# A_T_W_inv_A = A.T @ W_inv @ A
#
# # 计算(V^-1 + A^T * W^-1 * A)的逆矩阵
# V_inv_plus_A_T_W_inv_A_inv = inv(V_inv + A_T_W_inv_A)
#
# # 计算V^-1 * t_hat
# V_inv_t_hat = V_inv @ OD_vector
#
# # 计算A^T * W^-1 * f
# A_T_W_inv_f = A.T @ W_inv @ link_flows_vector
#
# # 步骤4: 计算O-D矩阵的估计值
# t_GLS = V_inv_plus_A_T_W_inv_A_inv @ (V_inv_t_hat + A_T_W_inv_f)
#
# # 输出结果
# print("优化后的O-D矩阵估计值 t_GLS:")
# print(t_GLS.reshape(N, N))  # 将结果reshape回N x N矩阵
#
# # 步骤5: 评估估计器的性能
# # 假设真实O-D矩阵
# t_true = OD_vector
#
# # 计算均方误差 (MSE)
# mse = np.mean((t_GLS - t_true) ** 2)
# print("均方误差 (MSE):", mse)