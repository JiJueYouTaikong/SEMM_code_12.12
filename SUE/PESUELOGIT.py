import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_utils.计算PESUELOGIT需要的M和D import init_M_D_matrices,create_graph
import functools
print = functools.partial(print, flush=True)
# 展平时变OD
def flatten_od(matrix):
    T, N, _ = matrix.shape
    result = np.zeros((T, N * N - N))
    for i in range(T):
        current_matrix = matrix[i]
        non_diagonal = current_matrix[~np.eye(N, dtype=bool)]
        result[i] = non_diagonal
    return result

# 展平历史OD
def flatten_his_od(matrix):
    N, _ = matrix.shape
    result = np.zeros((N * N - N))
    current_matrix = matrix
    non_diagonal = current_matrix[~np.eye(N, dtype=bool)]
    result = non_diagonal
    return result

def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    # 计算 MAPE，避免除以零
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(
            torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()



# def bpr_performance_function(x_hat, alpha, beta):
#     return alpha * (1 + beta * (x_hat ** 4))
# BPR性能函数
def bpr_t(t_min, x_max, x_hat_i, alpha, beta):
    """
    BPR函数
    :param t_min: 形状为[链路总数]的numpy数组，代表t的最小值向量
    :param x_hat_i: 形状为[链路总数]的numpy数组，代表估计的x_i向量
    :param x_max: 形状为[链路总数]的numpy数组，代表x的最大值向量
    :param alpha: 初始值为0.15，形状为[链路总数]的numpy数组代表各个链路的α值
    :param beta: 初始值为4， 形状为[链路总数]的numpy数组代表各个链路的β值
    :return: 形状为[链路总数]的numpy数组，计算得到的t_i向量
    """
    # 逐元素相乘和幂运算
    term = (1 + alpha * (x_hat_i / x_max)) ** beta
    t_i = t_min * term
    return t_i


# 定义模型
class TrafficModel(nn.Module):
    def __init__(self, num_links, num_paths, num_od_pairs, D, M, q_his):
        super(TrafficModel, self).__init__()
        # 初始化链路流量参数向量
        self.x_hat = nn.Parameter(torch.zeros(num_links))
        # 初始化BPR函数参数
        self.alpha = nn.Parameter(torch.full((num_links,), 0.15))
        self.beta = nn.Parameter(torch.full((num_links,), 4.0))
        # 初始化OD对向量
        mean_q_his = torch.mean(q_his)
        noise = torch.normal(mean=torch.zeros_like(q_his), std=0.1 * mean_q_his)
        self.q_hat = nn.Parameter(q_his + noise).float()
        self.D = torch.tensor(D, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

    def forward(self):
        # 步骤2：应用BPR性能函数得到旅行时间
        t = bpr_t(t_min,x_max,self.x_hat, self.alpha, self.beta)

        # 步骤3：计算路径效用
        path_utilities = torch.matmul(self.D.T, t)

        # 步骤4：计算路径选择概率
        path_selection_probabilities = torch.zeros(self.M.shape[1])
        # 根据OD对选取路径并计算logit概率
        for i in range(self.M.shape[0]):  # 遍历所有的 OD 对
            path_indices = torch.where(self.M[i] == 1)[0]  # 提取元组的第一个元素作为索引
            if len(path_indices) > 0:
                # print("path_indices",path_indices)
                od_path_utilities = path_utilities[path_indices]
                p = F.softmax(od_path_utilities, dim=0)
                # print("概率 p", p)
                # for i in range(len(path_indices)):
                path_selection_probabilities[path_indices] = p

        p = path_selection_probabilities
        # print("概率p向量", p.shape)
        # print("概率p向量", p[:10])
        # 步骤 5：计算路径流量
        # print("M q_hat", self.M.shape, self.q_hat.shape)
        q_vector = torch.matmul(self.M.T, self.q_hat)
        # print("得到的路径向量", q_vector.shape)
        f = q_vector * p
        # 步骤 6：计算链路流量
        x = torch.matmul(f, self.D.T)
        return x, t, f, p


# 收敛性检查函数（公式19）
def convergence_check(x, x_hat, epsilon=1e-5):
    """
    收敛性检查
    :param x: 估计flow向量
    :param x_hat: 初始化flow向量
    :param epsilon: 收敛阈值，1e-5
    :return: 如果满足收敛条件返回True，否则返回False
    """
    # 计算公式中的分子，即向量1-范数
    numerator = torch.linalg.norm(x_hat - x, ord=1)
    # 计算公式中的分母，即向量x的1-范数
    denominator = torch.linalg.norm(x, ord=1)
    # 计算ρ_j
    rho_j = numerator / denominator if denominator != 0 else 0
    # 检查是否满足收敛条件
    return rho_j < epsilon



# 损失函数（公式15）
def loss_function(x, x_hat,x_obs,q_hat, q_his):

    loss_x = torch.linalg.norm(x_hat - x_obs) ** 2
    loss_e = torch.linalg.norm(x - x_hat) ** 2
    loss_q = torch.linalg.norm(q_hat - q_his) ** 2

    loss = loss_x + loss_e + loss_q
    return loss


# # 示例参数
# num_links = 6
# num_paths = 4
# num_od_pairs = 2
# D = [
#     [1, 1, 0, 0],
#     [0, 0, 1, 1],
#     [1, 0, 1, 0],
#     [0, 1, 0, 1],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0]
# ]
# M = [
#     [1, 1, 0, 0],
#     [0, 0, 1, 1]
# ]


# 构建图结构
adj = np.load("../data/adj110_3.17.npy")
dist = np.load("../data/dist110_3.17.npy")
OD = np.load("../data/OD_完整批处理_3.17_Final.npy")
OD=OD[0]
G = create_graph(adj, dist)

# 计算两个矩阵 M是OD对-路径关联矩阵=[OD对数，路径数] D是路径-链路关联矩阵=[链路数，路径数]
M, D = init_M_D_matrices(OD,adj,G)


print(f"M:{M.shape},D:{D.shape}")

num_links = D.shape[0]
num_paths = D.shape[1]
num_od_pairs = M.shape[0]

# BPR函数参数
t_min = torch.full((num_links,), 2)
x_max = torch.full((num_links,), 500)

# 日志
log_filename = f"log/PESUELOGIT.log"


x_obs = np.load("data/Link_flow_TL_MSA-SUE_logit3_6_14.npy")
q_obs = np.load("data/OD_完整批处理_3.17_Final.npy")
T = q_obs.shape[0]
print(f"数据集的总时间步:{T}")
num_samples = 35

# OD对历史观测值
q_his = q_obs[:T-num_samples]             # [T_his,N,N]
q_his = np.mean(q_his, axis=0)
q_his = torch.from_numpy(flatten_his_od(q_his)) # [N*N-1]

# OD对观测值
q_obs = q_obs[-num_samples:]  # [T_test,N,N]
q_obs = flatten_od(q_obs)     # [T_test, N*N-1]
q_obs = torch.from_numpy(q_obs)
# 链路观测值
x_obs = x_obs[-num_samples:]  # [T_test,L]
x_obs = torch.from_numpy(x_obs)

print(f"历史OD对:{q_his.shape},时变OD对观测值:{q_obs.shape},时变链路流量观测值:{x_obs.shape}")

# 训练循环
max_iter = 2000
lrate = 0.1
total_rmse = 0
total_mae = 0
total_mape = 0


# 保存预测OD向量列表
all_q_hat_list = []


for i in range(num_samples):
    # 创建模型
    model = TrafficModel(num_links, num_paths, num_od_pairs, D, M, q_his)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    for j in range(max_iter):
        x_hat = model.x_hat.detach().clone()
        x, t, f, p = model()
        # 步骤7：收敛性检查
        if convergence_check(x, x_hat):
            print(f"Converged at iteration {j}")
            break
        # 步骤8：计算损失并反向传播
        loss = loss_function(x, model.x_hat,x_obs[i], model.q_hat, q_his)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"样本{i}, epoch{j}, Loss: {loss.item():.4f}")

    # 打印与评估
    pred_q_hat = model.q_hat.detach().cpu().numpy()
    all_q_hat_list.append(pred_q_hat)  # 收集当前预测OD向量


    print(f"样本{i}的真实OD Sum",int(sum(q_obs[i])))
    print(f"样本{i}的估计OD Sum",int(sum(model.q_hat)))

    rmse, mae, mape = calculate_rmse_mae(q_obs[i],model.q_hat)
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}")
    total_rmse += rmse
    total_mae += mae
    total_mape += mape


# 保存所有样本预测的 q_hat
q_hat_array = np.stack(all_q_hat_list, axis=0).astype(np.float32)  # shape: [num_samples, num_od_pairs]
np.save('../可视化/测试集TNN/Pred-PESL.npy', q_hat_array)
print("所有预测的q_hat已保存为Pred-PESL")

total_rmse = total_rmse / num_samples
total_mae = total_mae / num_samples
total_mape = total_mape / num_samples

print(f"最终RMSE:{total_rmse:.4f} MAE:{total_mae:.4f} MAPE:{total_mape:.4f}")

with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Lr={lrate},Iter={max_iter},Samples={num_samples} Final RMSE: {total_rmse:.4f} MAE: {total_mae:.4f} MAPE: {total_mape:.4f}\n")
