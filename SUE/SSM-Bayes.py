import time
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.nn.functional as F
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.util import NotUniqueError
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import torch.nn as nn

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
    new_flows = torch.zeros((N, N), dtype=torch.float32)  # 不需要梯度

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
                    assigned_flow = demand * prob.item()
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        new_flows[u, v] += assigned_flow

    # 更新路段的通行时间
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
        flow_results += flow_matrix

    return flow_results


class BayesianODEstimator:
    def __init__(self, N, adj_matrix, dist_matrix, observed_flow, init_od=None):
        self.N = N
        self.adj_matrix = adj_matrix
        self.dist_matrix = dist_matrix
        self.observed_flow = observed_flow
        self.best_loss = float('inf')
        self.best_od = None

        # 初始化OD矩阵参数范围
        if init_od is not None:
            self.init_od = init_od
            # 设置参数范围为初始值的±50%
            self.pbounds = {f'od_{i}_{j}': (0.8 * init_od[i, j], 1.2 * init_od[i, j])
                            for i in range(N) for j in range(N) if init_od[i, j] > 0}
        else:
            # 如果没有初始值，设置一个宽范围
            self.pbounds = {f'od_{i}_{j}': (0, 1000) for i in range(N) for j in range(N)}

    def evaluate_od(self, **od_params):
        # 将参数字典转换为OD矩阵
        od_matrix = np.zeros((self.N, self.N))
        for key, value in od_params.items():
            i, j = map(int, key.split('_')[1:])
            od_matrix[i, j] = value

        # 运行DTA模型
        flow_pred = run_dta_logit(
            torch.tensor(od_matrix, dtype=torch.float32),
            self.adj_matrix,
            self.dist_matrix,
            lambda_param,
            max_iter,
            tol,
            capacity
        )


        # # # 定义MSE损失函数
        # mse_loss = nn.MSELoss()
        # # 计算损失
        # loss = mse_loss(
        #     flow_pred,
        #     torch.tensor(self.observed_flow, dtype=torch.float32)
        # )

        # 计算损失
        loss = traffic_volume_loss(
           flow_pred,
           torch.tensor(self.observed_flow, dtype=torch.float32))

        # 保存最佳结果
        if loss < self.best_loss:
            self.best_loss = loss
        self.best_od = od_matrix.copy()

        # 贝叶斯优化需要最大化目标函数，所以我们返回负损失
        return -loss.item()


# 加载数据
adj_matrix = np.load('../data/adj110_3.17.npy')
dist_matrix = np.load('../data/dist110_3.17.npy')

# 数据集
N = 110
num_samples = 35
flow = np.load('data/Link_flow_TNN_3.19_可微.npy')
od = np.load('data/OD_完整批处理_3.17_Final.npy')
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)

# 参数
num_epochs = 50  # 贝叶斯优化迭代次数
tol = 0.05  # 收敛阈值
init_matrix = np.load('../data/初始OD估计_NN_25.3.18.npy')

# 结果记录
rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0

start_time = time.time()

for i in range(num_samples):
    print(f"\nProcessing sample {i + 1}/{num_samples}",flush=True)

    # 初始化贝叶斯优化器
    estimator = BayesianODEstimator(N, adj_matrix, dist_matrix, flow[i], init_matrix)

    # 设置优化器参数
    optimizer = BayesianOptimization(
        f=estimator.evaluate_od,
        pbounds=estimator.pbounds,
        random_state=42,
        verbose=2
    )

    # 设置日志记录
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # 运行优化
    optimizer.maximize(
        init_points=10,  # 初始随机探索点
        n_iter=num_epochs,  # 优化迭代次数
    )

    # 获取最佳结果
    best_od = estimator.best_od
    best_loss = estimator.best_loss

    # 计算评估指标
    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    pred_od_tensor = torch.tensor(best_od, dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od_tensor, od_tensor)

    # 记录结果
    loss_total += best_loss
    rmse_total += rmse
    mae_total += mae
    mape_total += mape

    print(f"Sample {i} - Best Loss: {best_loss:.4f}",flush=True)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}",flush=True)

    # # 可视化结果
    # if i % 1 == 0:  # 对每个样本都可视化
    #     vmin = min(best_od.min(), od[i].min())
    #     vmax = max(best_od.max(), od[i].max())
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     im1 = axes[0].imshow(best_od, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     axes[0].set_title(f'Estimated OD (Sample {i + 1})')
    #     fig.colorbar(im1, ax=axes[0])
    #
    #     im2 = axes[1].imshow(od[i], cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    #     axes[1].set_title(f'True OD (Sample {i + 1})')
    #     fig.colorbar(im2, ax=axes[1])
    #
    #     plt.show()

# 计算平均评估指标
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"\nFinal Results:",flush=True)
print(f"Average Test Loss: {loss_test:.4f}",flush=True)
print(f"RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}",flush=True)

end_time = time.time()
time_difference_seconds = end_time - start_time
times = time_difference_seconds / 60
print(f"Total time: {times:.2f} minutes",flush=True)

# 保存结果日志
with open("log/SSM-Bayes.log", "a") as f:
    f.write(f"Samples: {num_samples}, Iterations: {num_epochs}, Loss: {loss_test:.4f}, "
            f"RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}, "
            f"Time: {times:.2f} min\n")