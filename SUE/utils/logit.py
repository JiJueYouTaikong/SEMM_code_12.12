import numpy as np

# # 加载数据
# od_matrix = np.load('data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # [T, N, N]
# print(od_matrix[0,:10,:10])

adj_matrix = np.load('data/adj110.npy')  # [N, N]
# print(adj_matrix[:6,:6])

dist_matrix = np.load('data/dist110.npy')  # [N, N]
# print(dist_matrix[:6,:6])


# 参数定义
T, N, _ = od_matrix.shape


# Logit 模型概率计算函数
def logit_probability(travel_time, theta=1.0):
    exp_neg_cost = np.exp(-theta * travel_time)
    probabilities = exp_neg_cost / exp_neg_cost.sum(axis=1, keepdims=True)
    return probabilities

def logit(od):

    N, N = od.shape

    alpha = 0.15  # BPR 函数参数
    beta = 4  # BPR 函数参数
    capacity = 1000  # 每条路段的基础容量
    travel_time_base = 1  # 基础旅行时间（单位时间）

    # 计算路段阻抗（初始值为基础旅行时间）
    travel_time = np.ones((N, N)) * travel_time_base
    travel_time[adj_matrix == 0] = np.inf  # 非直接连通的路段设为无穷大

    link_flow = np.zeros((N, N))  # 初始路段流量矩阵

    # 1. 计算路径选择概率（基于 Logit 模型）
    probabilities = logit_probability(travel_time)

    # 2. 分配 OD 流量到路段
    for i in range(N):
        for j in range(N):
            if od[i, j] > 0:  # 有需求的 OD 对
                od_flow = od[i, j]
                link_flow[i] += od_flow * probabilities[i, j]

    # 3. 更新路段阻抗（基于流量和 BPR 函数）
    travel_time = travel_time_base * (1 + alpha * (link_flow_t / capacity) ** beta)
    travel_time[adj_matrix == 0] = np.inf  # 非连通路段仍为无穷大

    # 保存当前时间步的路段流量
    link_flows[t] = link_flow_t

# 保存结果
np.save('处理后的data/Link_Flow_25.2.25最新TNN.npy', link_flows)
print(f"路段流量数据已生成，形状为: {link_flows.shape}")
print(od_matrix[0,:10,:10])
od_sum = np.sum(od_matrix)
print(od_sum)
print("---------------------------------")
print(link_flows[0,:10,:10])
link_sum = np.sum(link_flows)
print(link_sum)
