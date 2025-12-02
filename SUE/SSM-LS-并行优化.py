import time
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from scipy.optimize import least_squares
import multiprocessing as mp
from functools import partial

# 设置随机数种子
torch.manual_seed(42)
np.random.seed(42)


def traffic_volume_loss(estimated_volume, observed_volume):
    """保留原有的损失函数计算方式"""
    mask = observed_volume != 0
    valid_estimated = torch.masked_select(estimated_volume, mask)
    valid_observed = torch.masked_select(observed_volume, mask)
    diff_ratio = (valid_estimated - valid_observed) / valid_observed
    return torch.sum(torch.square(diff_ratio))


def calculate_rmse_mae(predictions, targets):
    """保留原有的评估指标计算"""
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    non_zero_mask = targets != 0
    mape = torch.mean(torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) /
                                targets[non_zero_mask])) if non_zero_mask.sum() > 0 else torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()


# 参数设置
capacity = 500
alpha = 0.15
beta = 4
lambda_param = 1
max_iter = 1


def bpr_function(t0, v, c, alpha=0.15, beta=4):
    return t0 * (1 + alpha * (v / c) ** beta)


def process_od_pair(args, G, lambda_param):
    """处理单个OD对的函数，优化路径搜索和计算"""
    origin, destination, demand = args
    try:
        # 限制路径搜索深度为3跳
        paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=2))
        if not paths:
            return None

        path_times = []
        for path in paths:
            time = sum(G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
            path_times.append(time)

        path_probs = F.softmax(-lambda_param * torch.tensor(path_times), dim=0)

        flow_contrib = {}
        for path, prob in zip(paths, path_probs):
            assigned_flow = demand * prob.item()
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                flow_contrib[edge] = flow_contrib.get(edge, 0) + assigned_flow

        return flow_contrib
    except Exception as e:
        print(f"Error processing OD pair {origin}-{destination}: {str(e)}")
        return None


def parallel_logit_assignment(G, od_matrix, lambda_param, dist_matrix, capacity):
    """并行交通分配函数，修复结果合并问题"""
    N = od_matrix.shape[0]
    new_flows = torch.zeros((N, N), dtype=torch.float32)

    # 准备OD对参数
    od_pairs = [(o, d, od_matrix[o, d].item())
                for o in range(N) for d in range(N)
                if od_matrix[o, d] > 0]

    # 使用进程池并行处理
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:  # 限制最大进程数
        results = pool.map(partial(process_od_pair, G=G, lambda_param=lambda_param), od_pairs)

    # 合并结果 - 修复了这里的处理逻辑
    for res in filter(None, results):  # 过滤掉None结果
        for (u, v), flow in res.items():
            new_flows[u, v] += flow

    # 更新路段通行时间
    for u, v in G.edges():
        G[u][v]['distance'] = bpr_function(dist_matrix[u, v], new_flows[u, v].item(), capacity)

    return new_flows


def run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol, capacity):
    """运行动态交通分配的主函数"""
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    for u, v in G.edges():
        G[u][v]['distance'] = dist_matrix[u, v]

    flow_results = torch.zeros_like(od_matrix)
    for _ in range(max_iter):
        flow_results += parallel_logit_assignment(G, od_matrix, lambda_param, dist_matrix, capacity)

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
        """前向模型计算，添加异常处理"""
        try:
            ode_matrix = ode_flat.reshape(self.ode_shape)
            flow_pred = run_dta_logit(torch.tensor(ode_matrix),
                                      self.adj_matrix,
                                      self.dist_matrix,
                                      self.lambda_param,
                                      self.max_iter,
                                      self.tol,
                                      self.capacity)
            return flow_pred.numpy().flatten()
        except Exception as e:
            print(f"Forward model error: {str(e)}")
            return np.zeros(self.N * self.N)

    def optimize(self, observed_flow, max_nfev=5):
        """优化最小二乘求解过程"""
        observed_flat = observed_flow.flatten()
        mask = observed_flat != 0
        valid_observed = observed_flat[mask]

        def residuals(ode_flat):
            pred = self.forward_model(ode_flat)[mask]
            return (pred - valid_observed) / np.maximum(valid_observed, 1e-6)  # 避免除以零

        # 优化求解器参数
        result = least_squares(
            residuals,
            self.init_ode,
            method='trf',
            max_nfev=max_nfev,
            ftol=1e-3,
            xtol=1e-3,
            loss='soft_l1',
            verbose=2
        )

        return result.x.reshape(self.ode_shape)


# 数据加载和主流程保持不变
if __name__ == '__main__':
    # 加载数据
    adj_matrix = np.load('../data/adj110_3.17.npy')
    dist_matrix = np.load('../data/dist110_3.17.npy')

    N = 110
    num_samples = 3
    flow = np.load('data/Link_flow_TNN_3.19_可微.npy')[-num_samples:].astype(np.float32)
    od = np.load('data/OD_完整批处理_3.17_Final.npy')[-num_samples:].astype(np.float32)
    init_matrix = np.load('data/初始OD估计_NN_25.3.18.npy')

    # 确保在Windows下使用multiprocessing的正确方式
    mp.freeze_support()

    # 评估指标
    metrics = {'rmse': 0, 'mae': 0, 'mape': 0, 'loss': 0}
    start_time = time.time()

    for i in range(num_samples):
        print(f"\nProcessing sample {i + 1}/{num_samples}")
        print(f"真实OD: {od[i].sum():.2f}, 初始ODE: {init_matrix.sum():.2f}")

        optimizer = LeastSquaresOptimizer(N, init_matrix, adj_matrix, dist_matrix,
                                          lambda_param, max_iter, tol=0.4, capacity=capacity)

        try:
            optimized_ode = optimizer.optimize(flow[i], max_nfev=5)
            final_flow = optimizer.forward_model(optimized_ode.flatten()).reshape(N, N)

            # 计算指标
            loss = np.sum(((final_flow[flow[i] != 0] - flow[i][flow[i] != 0]) /
                           np.maximum(flow[i][flow[i] != 0], 1e-6)) ** 2)

            rmse, mae, mape = calculate_rmse_mae(
                torch.tensor(optimized_ode),
                torch.tensor(od[i]))

            metrics['loss'] += loss
            metrics['rmse'] += rmse
            metrics['mae'] += mae
            metrics['mape'] += mape

            print(f"优化结果: ODE总和={optimized_ode.sum():.2f}, "
                  f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")

        except Exception as e:
            print(f"处理样本{i}时出错: {str(e)}")
            continue

    # 输出最终结果
    time_used = (time.time() - start_time) / 60
    print(f"\n平均结果 - Loss: {metrics['loss'] / num_samples:.4f}, "
          f"RMSE: {metrics['rmse'] / num_samples:.4f}, "
          f"MAE: {metrics['mae'] / num_samples:.4f}, "
          f"MAPE: {metrics['mape'] / num_samples:.4f}")
    print(f"总耗时: {time_used:.2f}分钟")