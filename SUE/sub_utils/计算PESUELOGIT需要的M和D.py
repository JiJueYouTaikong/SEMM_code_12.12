import networkx as nx
import numpy as np

def create_graph(adj, dist):
    G = nx.DiGraph()
    num_nodes = len(adj)
    # 添加节点
    for i in range(num_nodes):
        G.add_node(i)
    # 添加边，根据邻接矩阵和距离矩阵
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] == 1:
                G.add_edge(i, j, weight=dist[i][j])
    return G

def init_M_D_matrices(OD,adj,G):
    '''
    给定OD和图结构，返回M,D
    :param OD:
    :param G:
    :return: M,D [N*N-1,2*N*N-1]  [link，2*N*N-1]
    '''

    num_nodes = len(OD)

    # 确定 O - D 对集合 W
    W = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                W.append((i, j))
    # 确定路径集合 H 和 H_w，每个 O - D 对只取最短的两条路径
    H = []
    H_w = {w: [] for w in W}
    for w in W:
        print("xx")
        source, target = w
        all_paths = list(nx.all_simple_paths(G, source, target,cutoff=2))
        path_distances = []
        for path in all_paths:
            path_distance = 0
            for k in range(len(path) - 1):
                path_distance += G[path[k]][path[k + 1]]['weight']
            path_distances.append(path_distance)
        sorted_paths = [path for _, path in sorted(zip(path_distances, all_paths))]
        shortest_two_paths = sorted_paths[:2]
        H_w[w] = shortest_two_paths
        H.extend(shortest_two_paths)

    # 构建路径 - 需求关联矩阵 M
    M = np.zeros((len(W), len(W) * 2))
    for w_index, w in enumerate(W):
        for h_index, h in enumerate(H):
            if h in H_w[w]:
                M[w_index][h_index] = 1

    # 计算链路总数
    link_count = int(np.sum(adj))


    # 构建路径 - 链路关联矩阵 D
    D = np.zeros((link_count, len(W) * 2))
    link_index = 0
    link_mapping = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] == 1:
                link_mapping[(i, j)] = link_index
                link_index += 1
    for h_index, h in enumerate(H):
        for k in range(len(h) - 1):
            source, target = h[k], h[k + 1]
            link_idx = link_mapping[(source, target)]
            D[link_idx][h_index] = 1

    return M, D

#
# # 简单示例数据
# OD = np.array([
#     [0, 1, 0],
#     [1, 0, 1],
#     [0, 1, 0]
# ])
#
# adj = np.array([
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]
# ])
#
# dist = np.array([
#     [100, 1, 2],
#     [1, 100, 1],
#     [2, 1, 100]
# ])
#
# # 调用函数
# G =create_graph(adj,dist)
# M, D = init_M_D_matrices(OD, G)
#
# # 打印结果
# print("路径 - 需求关联矩阵 M:")
# print(M)
# print("M 的维度:", M.shape)
# print("路径 - 链路关联矩阵 D:")
# print(D)
# print("D 的维度:", D.shape)
