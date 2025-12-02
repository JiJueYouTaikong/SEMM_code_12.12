import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import torch.nn.functional as F
from matplotlib import pyplot as plt

# 设置随机数种子
torch.manual_seed(42)
np.random.seed(42)

def traffic_volume_loss(estimated_volume, observed_volume):
    """
    计算交通流量损失 F_{OD}，忽略观测值为0的位置
    :param estimated_volume: 估计的交通流量，形状为 [N, N] 的张量
    :param observed_volume: 观测的交通流量，形状为 [N, N] 的张量
    :return: 交通流量损失值
    """
    assert estimated_volume.shape == observed_volume.shape
    # 创建掩码，观测值不为0的位置为True，否则为False
    mask = observed_volume != 0
    # 根据掩码筛选出观测值不为0的估计流量和观测流量
    valid_estimated_volume = torch.masked_select(estimated_volume, mask)
    valid_observed_volume = torch.masked_select(observed_volume, mask)
    # 计算 (Dx - DMx) / DMx
    diff_ratio = (valid_estimated_volume - valid_observed_volume) / valid_observed_volume
    # 对结果平方
    squared_diff_ratio = torch.square(diff_ratio)
    # 对所有有效元素求和
    loss = torch.sum(squared_diff_ratio)
    return loss


# 计算 RMSE、MAE 和 MAPE
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


# 参数设置
capacity = 500  # 路段容量
alpha = 0.15  # BPR 函数的 alpha 参数
beta = 4  # BPR 函数的 beta 参数
lambda_param = 1  # Logit 模型灵敏度参数


# BPR 函数计算通行时间
def bpr(t0, v, c):
    '''
    :param t0: 路段初始通行时间
    :param v:  当前flow
    :param c:  max_cap
    :param alpha:
    :param beta:
    :return: 通行时间
    '''
    return t0 * (1 + alpha * (v / c) ** beta)

# Logit 模型分配流量
def logit_traffic_assignment(G, od_matrix_t, lambda_param):
    N = od_matrix_t.shape[0]
    temp_flows = torch.zeros((N, N), dtype=torch.float32, requires_grad=True)  # 存储流量矩阵

    temp_flows = temp_flows.clone()

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
                tt = sum(G[p[i]][p[i + 1]]['time'] for i in range(len(p) - 1))
                times.append(tt)

            # Logit Prob (PyTorch)
            t_tensor = torch.tensor(times)
            probs = F.softmax(-lambda_param * t_tensor, dim=0).numpy()

            # 分配
            for p, prob in zip(paths, probs):
                flow = demand * prob
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    temp_flows[u, v] = temp_flows[u, v] + flow  # 非原地

    return temp_flows

# 运行 DTA Logit 模型
def run_dta_logit(G,od_matrix, adj_matrix, dist_matrix, lambda_param):
    # 分配流量
    flow_matrix = logit_traffic_assignment(G, od_matrix, lambda_param)
    return flow_matrix

# 双层框架模型
class BilevelFramework(nn.Module):
    def __init__(self, N, init_matrix):
        super(BilevelFramework, self).__init__()
        # 初始化出行通量矩阵为可训练参数
        # self.ode = nn.Parameter(torch.randn(N, N))
        self.ode = nn.Parameter(torch.from_numpy(init_matrix).float())

    def forward(self, G,adj_matrix, dist_matrix, lambda_param):
        flow_pred = run_dta_logit(G,self.ode, adj_matrix, dist_matrix, lambda_param)
        return flow_pred



def initialize_graph(adj_matrix, dist_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for u, v in G.edges:
        G[u][v]['t0'] = dist_matrix[u, v]
        G[u][v]['time'] = dist_matrix[u, v]
        G[u][v]['flow'] = 0.0
    return G


# 加载数据
adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]

# 数据集
N = 110
num_samples = 2
flow = np.load('data/Link_flow_TNN_MSA-SUE_logit3_6_14.npy')  # 【注】观测的交通流量 19：2 / 23：3
od = np.load('data/OD_完整批处理_3.17_Final.npy')  # OD 矩阵
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)
print("6.15.1",flush=True)

# 500 20   [2000 5]   [1000, 0.01]

num_epochs = 40
max_iter = 20  # 最大迭代次数
msa_tol = 5e-3
tol = 2

rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0
learning_rate = 100

init_matrix = np.load('data/初始OD估计_NN_25.3.18.npy')

start_time = time.time()


# 初始化一个列表来保存所有预测的OD矩阵
all_predicted_ods = []

for i in range(num_samples):

    # 计算损失
    flow_tensor = torch.tensor(flow[i], dtype=torch.float32)

    # 初始化单个样本的初始模型和OD矩阵
    model = BilevelFramework(N,init_matrix)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("--------------------------",flush=True)
    print(f"真实OD:{od[i].sum():.2f}",flush=True)
    print(f"初始ODE:{model.ode.sum():2f}",flush=True)

    for epoch in range(num_epochs):

        optimizer.zero_grad()

        # 初始化单个样本的原始图结构
        G = initialize_graph(adj_matrix, dist_matrix)

        prev_flow = torch.zeros((N, N), dtype=torch.float32)

        for k in range(max_iter):

            # 前向传播
            new_flow = model(G,adj_matrix, dist_matrix, lambda_param)

            # MSA update
            step = 1 / (k + 1)
            flow_est = (1 - step) * prev_flow + step * new_flow

            # 收敛判断
            diff = torch.abs(flow_est - prev_flow).max()
            print(f"    Max diff: {diff:.6f}")
            if diff < msa_tol:
                print("    Converged.")
                break

            prev_flow = flow_est.clone()

            # 更新图上 link flow & travel time
            for u, v in G.edges:
                G[u][v]['flow'] = flow_est[u, v]
                G[u][v]['time'] = bpr(G[u][v]['t0'], G[u][v]['flow'], capacity)


        loss = criterion(flow_est, flow_tensor)

        if loss <= tol:
            print(f"Loss is best:{loss}",flush=True)
            break

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch, num_epochs, loss.item()),flush=True)
            print(f"ODE:{model.ode.sum():.2f}, T: {od[i].sum():.2f}", flush=True)

    # 记录损失和评估指标
    loss_total += loss.item()
    pred_od = model.ode.detach()

    # 将当前预测的OD矩阵添加到列表中
    all_predicted_ods.append(pred_od.numpy())

    print(f"最终ODE:{model.ode.sum():.2f}",flush=True)
    print(f"真实OD:{od[i].sum():.2f}",flush=True)


    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od, od_tensor)
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}",flush=True)
    rmse_total += rmse
    mae_total += mae
    mape_total += mape


# 计算平均评估指标
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

# 保存所有预测的OD矩阵到一个npy文件
np.save('../可视化/测试集TNN/Pred_SSM_SUE_MSA_6.15.npy', np.array(all_predicted_ods))
print(f"所有预测的OD矩阵已保存",flush=True)

print(f"Total Test loss: {loss_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}",flush=True)

end_time = time.time()
time_difference_seconds = end_time - start_time
times = time_difference_seconds / 60
print(f"耗时{times:.2f}min",flush=True)

log_filename = f"log/SSM.log"
with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Samples: {num_samples}, Epochs: {num_epochs}, Lr: {learning_rate}, Tol: {tol}, Test Loss: {loss_test:.4f} RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}, Times: {times}\n")

