import numpy as np

# 定义参考OD矩阵 T0 (N=8)
N = 110
# T0 = np.random.rand(N, N) * 100  # 随机生成一个8x8的参考OD矩阵，值在0到100之间
od = np.load('../data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]
T0 = od[:100]
non_zero = np.count_nonzero(T0)
print(non_zero)
print(T0.shape)


# 蒙特卡洛模拟生成训练数据
num_samples = 86  # 生成1000个样本
T_samples = []  # 存储生成的OD矩阵
X_samples = []  # 存储对应的链路流量

list_all = []
for i in range(num_samples):

    # print(f"开始采样蒙特卡洛第{i}组数据")

    # print("---------------------")
    for j in range(T0.shape[0]):

        # print(f"{i} --> 为第{j}个时间步生成OD")

        # 定义总需求的均值和标准差
        mu_D = np.sum(T0[j])  # 总需求的均值
        sigma_D = mu_D * 0.1  # 总需求的标准差，假设为均值的10%

        # 定义随机波动的标准差
        sigma_w = 0.25 * T0[j]  # 随机波动的标准差，假设为T0的25%


        # 生成总需求 D
        D = np.random.normal(mu_D, sigma_D)


        # 生成随机波动 gamma
        gamma = np.random.normal(0, sigma_w)

        # print(D, gamma)

        # 生成随机的OD矩阵 T
        F = T0[j] / mu_D  # 归一化参考需求矩阵
        T = D * F + gamma

        # 将T加入样本集
        T_samples.append(T)
        # print("list长度：",len(T_samples))

    # list_all = np.concatenate(T_samples, axis=0)
    # print("list长度：", len(list_all))

# 将样本集转换为numpy数组
T_samples = np.array(T_samples)
T_samples = np.round(T_samples)


for i in range(T_samples.shape[0]):

    np.fill_diagonal(T_samples[i], 0)

# 打印生成的样本

print(T_samples[0,60:68, 60:68])
non_zero = np.count_nonzero(T_samples)
print("蒙特卡洛生成的总OD矩阵，非零值数量",non_zero)
max = T_samples.max()
print("蒙特卡洛生成的总OD矩阵，最大值",max)
# print(X_samples[0])
print("蒙特卡洛生成的总OD矩阵shape:", T_samples.shape)

OD_total = np.concatenate((T_samples, od), axis=0)
print(OD_total.shape)

np.save('../data/仿真_真实_总OD矩阵_V1.2_25.2.26', OD_total)




# Logit 模型概率计算函数
def logit_probability(travel_time, theta=1.0):
    exp_neg_cost = np.exp(-theta * travel_time)
    probabilities = exp_neg_cost / exp_neg_cost.sum(axis=1, keepdims=True)
    return probabilities


def sue_traffic_assignment(od_matrix, adj_matrix, distance_matrix, theta=0.5, epsilon=1e-6, max_iter=1000):
    """
    基于相继平均法（MSA）的时变随机用户均衡（SUE）交通分配函数

    参数:
    od_matrix (np.ndarray): 时变的OD矩阵，形状为 (T, N, N)，T 为时间步数，N 为节点数量
    adjacency_matrix (np.ndarray): 基于距离的邻接矩阵，形状为 (N, N)
    distance_matrix (np.ndarray): 存储距离的矩阵，形状为 (N, N)
    theta (float): Logit模型的离散参数
    epsilon (float): 收敛阈值
    max_iter (int): 最大迭代次数

    返回:
    np.ndarray: 每个时间步的路段流量矩阵，形状为 (T, N, N)
    """
    T, N = od_matrix.shape[0], od_matrix.shape[1]

    alpha = 0.15  # BPR 函数参数
    beta = 4  # BPR 函数参数
    capacity = 1000  # 每条路段的基础容量
    travel_time_base = 1  # 基础旅行时间（单位时间）

    # 初始化路段流量矩阵 [T, N, N]
    link_flows = np.zeros((T, N, N))

    # 计算路段阻抗（初始值为基础旅行时间）
    travel_time = np.ones((N, N)) * travel_time_base
    travel_time[adj_matrix == 0] = np.inf  # 非直接连通的路段设为无穷大


    # 动态交通分配 (DTA)
    for t in range(T):  # 遍历每个时间步
        od_t = od_matrix[t]  # 当前时间步的 OD 矩阵
        link_flow_t = np.zeros((N, N))  # 当前时间步的路段流量矩阵

        # 1. 计算路径选择概率（基于 Logit 模型）
        probabilities = logit_probability(travel_time)

        # 2. 分配 OD 流量到路段
        for i in range(N):
            for j in range(N):
                if od_t[i, j] > 0:  # 有需求的 OD 对
                    od_flow = od_t[i, j]
                    link_flow_t[i] += od_flow * probabilities[i, j]

        # 3. 更新路段阻抗（基于流量和 BPR 函数）
        travel_time = travel_time_base * (1 + alpha * (link_flow_t / capacity) ** beta)
        travel_time[adj_matrix == 0] = np.inf  # 非连通路段仍为无穷大

        # 保存当前时间步的路段流量
        link_flows[t] = link_flow_t

    # 保存结果
    # np.save('处理后的data/road_segment_flows.npy', link_flows)
    return link_flows


od_matrix = T_samples
speed = np.load('../data/武汉速度数据集_1KM_110区域_25.1.14.npy')
speed = speed[:,:,0]
speed_data = speed[:100]

# 初始化一个空列表来存储扰动后的数据集
perturbed_data_list = []

# 进行86次噪声扰动
for _ in range(86):
    # 生成与速度数据集相同形状的随机噪声，均值为0，标准差为10
    noise = np.random.normal(loc=0, scale=5, size=speed_data.shape)

    # 将噪声添加到速度数据集上
    perturbed_data = speed_data + noise

    # 将扰动后的数据集添加到列表中
    perturbed_data_list.append(perturbed_data)

# 将列表中的所有数据集合并成一个 [86*T, N] 的数据集
final_data = np.vstack(perturbed_data_list)




final_data[final_data < 0] = 0

print("平均速度矩阵",final_data.shape)
# print(average_speeds[:10, :30])


Speed_total = np.concatenate((final_data, speed), axis=0)
print(Speed_total.shape)
np.save('../data/仿真_真实_总Speed矩阵_V1.2_25.2.26', Speed_total)