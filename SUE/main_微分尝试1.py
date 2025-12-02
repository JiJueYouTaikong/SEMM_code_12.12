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
max_iter = 1  # 最大迭代次数


# BPR 函数计算通行时间
def bpr_function(t0, v, c, alpha=0.15, beta=4):
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
def logit_traffic_assignment(G, od_matrix_t, lambda_param, dist_matrix, capacity):
    N = od_matrix_t.shape[0]
    new_flows = torch.zeros((N, N), dtype=torch.float32, requires_grad=True)  # 存储当前时间步的流量矩阵

    for origin in range(N):
        # print(origin)
        for destination in range(N):
            if od_matrix_t[origin, destination] > 0:
                demand = od_matrix_t[origin, destination]

                # 计算从起点到目的地的路径通行时间
                paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=2))
                if not paths:  # 无路径跳过
                    continue

                path_times = []
                for path in paths:
                    time = sum(G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                    path_times.append(time)

                # 使用 PyTorch 计算 Logit 模型选择概率
                path_times_tensor = torch.tensor(path_times, dtype=torch.float32, requires_grad=True)
                path_probs = F.softmax(-lambda_param * path_times_tensor, dim=0)

                # 将流量按概率分配到路径
                for path, prob in zip(paths, path_probs):
                    assigned_flow = demand * prob
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # 避免 in-place 操作
                        new_flows = new_flows.clone()  # 创建副本以避免 in-place 操作
                        new_flows[u, v] = new_flows[u, v] + assigned_flow



    # 更新路段的通行时间
    with torch.no_grad():  # 不需要计算梯度
        for u, v in G.edges:
            flow = new_flows[u, v]
            t0 = dist_matrix[u, v]
            G[u][v]['distance'] = bpr_function(t0, flow, capacity)

    return new_flows

# 运行 DTA Logit 模型
def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity):
    N, _ = od_matrix.shape

    # 初始化图
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for i, j in G.edges:
        G[i][j]['distance'] = dist_matrix[i, j]
        G[i][j]['flow'] = 0  # 初始流量为 0

    flow_results = torch.zeros((N, N), dtype=torch.float32, requires_grad=True)  # 存储流量矩阵

    for iteration in range(max_iter):
        # 分配流量
        flow_matrix = logit_traffic_assignment(G, od_matrix, lambda_param, dist_matrix, capacity)
        flow_results = flow_results + flow_matrix

    return flow_results

# 双层框架模型
class BilevelFramework(nn.Module):
    def __init__(self, N, init_matrix):
        super(BilevelFramework, self).__init__()
        # 初始化出行通量矩阵为可训练参数
        # self.ode = nn.Parameter(torch.randn(N, N))
        self.ode = nn.Parameter(torch.from_numpy(init_matrix).float())

    def forward(self, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity):
        flow_pred = run_dta_logit(self.ode, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity)
        return flow_pred




# 加载数据
adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]

# 数据集
N = 110
num_samples = 35
flow = np.load('data/Link_flow_TNN_3.19_可微.npy')  # 观测的交通流量 19：2 / 23：3
od = np.load('data/OD_完整批处理_3.17_Final.npy')  # OD 矩阵
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)


# 500 20   [2000 5]

num_epochs = 1000
tol =0.01  # 收敛阈值

rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0
learning_rate = 40

init_matrix = np.load('data/初始OD估计_NN_25.3.18.npy')

for i in range(num_samples):
    # 对每一组样本随机初始化估计的 OD 矩阵
    model = BilevelFramework(N,init_matrix)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("--------------------------")
    print(f"真实OD:{od[i].sum():.2f}")
    print(f"初始ODE:{model.ode.sum():2f}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        print(f"ODE:{model.ode.sum():.2f}, T: {od[i].sum():.2f}")

        # 前向传播
        predicted_flow = model(adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity)

        # 计算损失
        flow_tensor = torch.tensor(flow[i], dtype=torch.float32)
        loss = criterion(predicted_flow, flow_tensor)

        if loss <= tol:
            break

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # 记录损失和评估指标
    loss_total += loss.item()
    pred_od = model.ode.detach()
    print(f"最终ODE:{model.ode.sum():.2f}")
    print(f"真实OD:{od[i].sum():.2f}")


    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od, od_tensor)
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}")
    rmse_total += rmse
    mae_total += mae
    mape_total += mape

    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制最终 OD 估计矩阵热力图
    im1 = axes[0].imshow(pred_od.numpy(), cmap='Blues', interpolation='nearest')
    axes[0].set_title(f'Final Estimated OD Matrix (Sample {i + 1})')
    fig.colorbar(im1, ax=axes[0])

    # 绘制真实 OD 矩阵热力图
    im2 = axes[1].imshow(od_tensor.numpy(), cmap='Blues', interpolation='nearest')
    axes[1].set_title(f'True OD Matrix (Sample {i + 1})')
    fig.colorbar(im2, ax=axes[1])

    plt.show()

# 计算平均评估指标
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"Total Test loss: {loss_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}")

log_filename = f"log/SUE_G.log"
with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Samples: {num_samples}, Epochs: {num_epochs}, Lr: {learning_rate}, Tol: {tol}, Test Loss: {loss_test:.4f} RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}\n")

