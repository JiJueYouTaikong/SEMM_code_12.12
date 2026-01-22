import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# ============================
# === 数据加载 ===============
# ============================

od_matrix = np.load('data/OD_完整批处理_3.17_Final.npy')  # [T, N, N]
adj_matrix = np.load('data/adj110_3.17.npy')              # [N, N]
dist_matrix = np.load('data/dist110_3.17.npy')            # [N, N]

od_matrix = od_matrix[-35:]


# # 简单数据
# od_matrix = np.array([
#     [0, 30, 10, 0],
#     [0, 0, 50, 10],
#     [0, 0, 0, 5],
#     [0, 0, 0, 0]
# ])
#
# od_matrix = np.expand_dims(od_matrix,axis=0)
#
# adj_matrix = np.array([
#     [0, 1, 0, 0],
#     [1, 0, 1, 1],
#     [0, 1, 0, 1],
#     [0, 1, 1, 0]
# ])
# dist_matrix = np.array([
#     [0, 1, 30, 30],
#     [1, 0, 2, 1],
#     [30, 2, 0, 1],
#     [30, 1, 1, 0]
# ])

print("邻接矩阵非零值:", np.count_nonzero(adj_matrix))

# ============================
# === 参数 ===================
# ============================

capacity = 500
alpha = 0.15
beta = 4
lambda_param = 1
max_iter = 20    # 迭代次数
tol = 1e-3

# ============================
# === 初始化图结构 ============
# ============================

def initialize_graph(adj_matrix, dist_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for u, v in G.edges:
        G[u][v]['t0'] = dist_matrix[u, v]
        G[u][v]['time'] = dist_matrix[u, v]
        G[u][v]['flow'] = 0.0
    return G

# ============================
# === BPR 函数 ===============
# ============================

def bpr(t0, v, c):
    return t0 * (1 + alpha * (v / c) ** beta)

# ============================
# === Logit 路径分配 ==========
# ============================

def logit_assignment(G, od_matrix_t, lambda_param):
    N = od_matrix_t.shape[0]
    temp_flows = np.zeros((N, N))  # link flow

    for origin in range(N):
        for dest in range(N):
            demand = od_matrix_t[origin, dest]
            if demand <= 0 or origin == dest:
                continue

            # 生成所有可行路径 (可调 cutoff)
            paths = list(nx.all_simple_paths(G, source=origin, target=dest, cutoff=3))
            if not paths:
                continue

            # 路径 travel time
            times = []
            for p in paths:
                tt = sum(G[p[i]][p[i+1]]['time'] for i in range(len(p)-1))
                times.append(tt)

            # Logit Prob (PyTorch)
            t_tensor = torch.tensor(times)
            probs = F.softmax(-lambda_param * t_tensor, dim=0).numpy()

            # 分配
            for p, prob in zip(paths, probs):
                flow = demand * prob
                for i in range(len(p)-1):
                    u, v = p[i], p[i+1]
                    temp_flows[u, v] += flow

    return temp_flows

# ============================
# === MSA-SUE 主流程 ==========
# ============================

def msa_sue(od_matrix, adj_matrix, dist_matrix, max_iter, tol):
    T, N, _ = od_matrix.shape
    flow_results = np.zeros((T, N, N))

    for t in range(T):
        print(f"\n==== Time step {t+1}/{T} ====")
        od_matrix_t = od_matrix[t]
        G = initialize_graph(adj_matrix, dist_matrix)
        prev_flow = np.zeros((N, N))
        for k in range(max_iter):
            print(f"  Iteration {k+1}")
            # 当前路径分配
            new_flow = logit_assignment(G, od_matrix_t, lambda_param)
            # MSA update
            step = 1 / (k + 1)
            flow = (1 - step) * prev_flow + step * new_flow
            # 收敛判断
            diff = np.abs(flow - prev_flow).max()
            print(f"    Max diff: {diff:.6f}")
            if diff < tol:
                print("    Converged.")
                break
            prev_flow = flow.copy()
            # 更新图上 link flow & travel time
            for u, v in G.edges:
                G[u][v]['flow'] = flow[u, v]
                G[u][v]['time'] = bpr(G[u][v]['t0'], G[u][v]['flow'], capacity)

        flow_results[t] = flow
    return flow_results

# ============================
# === 可视化 =================
# ============================

def visualize_graph(G, iteration, timestep):
    pos = nx.spring_layout(G, seed=42)
    edge_colors = [G[u][v]['flow'] for u, v in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.viridis, width=2)
    plt.title(f"Time {timestep} - Iter {iteration}")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(edge_colors)
    plt.colorbar(sm, label='Flow')
    plt.show()

# ============================
# === 运行 ===================
# ============================

flow_results = msa_sue(od_matrix, adj_matrix, dist_matrix, max_iter, tol)

np.save('处理后的data_6_14/Link_flow_TNN_MSA-SUE_logit3_6_14.npy', flow_results)
print("✅ MSA-SUE done. Result saved.")
print(flow_results.shape)
print(f"Total OD: {np.sum(od_matrix)}")
print(f"Total assigned flow: {np.sum(flow_results)}")
