import time
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

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


def bpr_function(t0, v, c, alpha=0.15, beta=4):
    return t0 * (1 + alpha * (v / c) ** beta)


def logit_traffic_assignment(G, od_matrix_t, lambda_param, dist_matrix, capacity):
    N = od_matrix_t.shape[0]
    new_flows = torch.zeros((N, N), dtype=torch.float32)

    for origin in range(N):
        for destination in range(N):
            if od_matrix_t[origin, destination] > 0:
                demand = od_matrix_t[origin, destination]
                paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=2))
                if not paths:
                    continue

                path_times = []
                for path in paths:
                    time = sum(G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
                    path_times.append(time)

                path_times_tensor = torch.tensor(path_times, dtype=torch.float32)
                path_probs = F.softmax(-lambda_param * path_times_tensor, dim=0)

                for path, prob in zip(paths, path_probs):
                    assigned_flow = demand * prob
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        new_flows[u, v] = new_flows[u, v] + assigned_flow

    # 更新路段的通行时间
    with torch.no_grad():
        for u, v in G.edges:
            flow = new_flows[u, v]
            t0 = dist_matrix[u, v]
            G[u][v]['distance'] = bpr_function(t0, flow, capacity)

    return new_flows


def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity):
    N, _ = od_matrix.shape
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for i, j in G.edges:
        G[i][j]['distance'] = dist_matrix[i, j]
        G[i][j]['flow'] = 0

    flow_results = torch.zeros((N, N), dtype=torch.float32)

    for iteration in range(max_iter):
        flow_matrix = logit_traffic_assignment(G, od_matrix, lambda_param, dist_matrix, capacity)
        flow_results = flow_results + flow_matrix

    return flow_results


class LeastSquaresOptimizer:
    def __init__(self, N, init_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity):
        self.N = N
        self.ode_shape = (N, N)
        self.init_ode = init_matrix.flatten()
        self.adj_matrix = adj_matrix
        self.dist_matrix = dist_matrix
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.capacity = capacity

    def forward_model(self, ode_flat):
        ode_matrix = ode_flat.reshape(self.ode_shape)
        flow_pred = run_dta_logit(torch.tensor(ode_matrix),
                                  self.adj_matrix,
                                  self.dist_matrix,
                                  self.lambda_param,
                                  self.max_iter,
                                  self.tol,
                                  self.capacity)
        return flow_pred.numpy().flatten()

    def optimize(self, observed_flow, max_nfev=100):
        observed_flow_flat = observed_flow.flatten()

        def residuals(ode_flat):
            pred_flow_flat = self.forward_model(ode_flat)
            mask = observed_flow_flat != 0
            res = (pred_flow_flat[mask] - observed_flow_flat[mask]) / observed_flow_flat[mask]
            return res

        result = least_squares(residuals,
                               self.init_ode,
                               method='trf',
                               max_nfev=max_nfev,
                               ftol=1e-2,  # 调小以要求更精确的解（但可能增加计算时间）
                               xtol=1e-2,  # 类似ftol但针对参数变化
                               gtol=1e-2,  # 控制梯度容忍度
                               x_scale='jac',  # 更好的缩放
                               tr_solver='lsmr',  # 更快的求解器
                               verbose=2)

        return result.x.reshape(self.ode_shape)


# 加载数据
adj_matrix = np.load('../data/adj110_3.17.npy')
dist_matrix = np.load('../data/dist110_3.17.npy')

# 数据集
N = 110
num_samples = 3
flow = np.load('data/Link_flow_TNN_3.19_可微.npy')
od = np.load('data/OD_完整批处理_3.17_Final.npy')
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)

num_epochs = None
tol = 0.4
init_matrix = np.load('data/初始OD估计_NN_25.3.18.npy')

rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0

start_time = time.time()

for i in range(num_samples):
    print("--------------------------")
    print(f"Processing sample {i + 1}/{num_samples}")
    print(f"真实OD:{od[i].sum():.2f}")
    print(f"初始ODE:{init_matrix.sum():.2f}")

    # 初始化优化器
    optimizer = LeastSquaresOptimizer(N, init_matrix, adj_matrix, dist_matrix,
                                      lambda_param, max_iter, tol, capacity)

    # 执行优化
    optimized_ode = optimizer.optimize(flow[i])

    # 计算最终流量预测
    final_flow = optimizer.forward_model(optimized_ode.flatten()).reshape(N, N)

    # 计算损失
    loss = np.sum(((final_flow[flow[i] != 0] - flow[i][flow[i] != 0]) / flow[i][flow[i] != 0]) ** 2)
    loss_total += loss

    # 计算评估指标
    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    pred_od_tensor = torch.tensor(optimized_ode, dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od_tensor, od_tensor)

    print(f"最终ODE:{optimized_ode.sum():.2f}")
    print(f"真实OD:{od[i].sum():.2f}")
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}")

    rmse_total += rmse
    mae_total += mae
    mape_total += mape

    # if i % 10 == 0:
    #     vmin = min(optimized_ode.min(), od[i].min())
    #     vmax = max(optimized_ode.max(), od[i].max())
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    #     im1 = axes[0].imshow(optimized_ode, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     axes[0].set_title(f'Final Estimated OD Matrix (Sample {i + 1})')
    #     fig.colorbar(im1, ax=axes[0])
    #
    #     im2 = axes[1].imshow(od[i], cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     axes[1].set_title(f'True OD Matrix (Sample {i + 1})')
    #     fig.colorbar(im2, ax=axes[1])
    #
    #     plt.show()

# 计算平均评估指标
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"Total Test loss: {loss_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}")

end_time = time.time()
time_difference_seconds = end_time - start_time
times = time_difference_seconds / 60
print(f"耗时{times:.2f}min")

log_filename = f"log/SUE_LeastSquares.log"
with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Samples: {num_samples}, MaxIter: {num_epochs}, Tol: {tol}, Test Loss: {loss_test:.4f} RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}, Times: {times}\n")
