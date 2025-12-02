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
    assert estimated_volume.shape == observed_volume.shape
    mask = observed_volume != 0
    valid_estimated_volume = torch.masked_select(estimated_volume, mask)
    valid_observed_volume = torch.masked_select(observed_volume, mask)
    diff_ratio = (valid_estimated_volume - valid_observed_volume) / valid_observed_volume
    squared_diff_ratio = torch.square(diff_ratio)
    loss = torch.sum(squared_diff_ratio)
    return loss


def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()


def bpr_function(t0, v, c, alpha=0.15, beta=4):
    return t0 * (1 + alpha * (v / c) ** beta)


def precompute_paths(G, max_path_len=2):
    od_paths = {}
    N = G.number_of_nodes()
    for origin in range(N):
        for destination in range(N):
            if origin != destination:
                paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=max_path_len))
                if paths:
                    od_paths[(origin, destination)] = paths
    return od_paths


def compute_path_times(od_paths, dist_matrix):
    path_times_dict = {}
    for (origin, destination), paths in od_paths.items():
        path_times = []
        for path in paths:
            time = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
            path_times.append(time)
        path_times_dict[(origin, destination)] = torch.tensor(path_times, dtype=torch.float32)
    return path_times_dict


def logit_traffic_assignment_fast(od_matrix_t, od_paths, path_times_dict, G, dist_matrix, capacity, lambda_param):
    N = od_matrix_t.shape[0]
    new_flows = torch.zeros((N, N), dtype=torch.float32)
    for (origin, destination), paths in od_paths.items():
        demand = od_matrix_t[origin, destination]
        if demand <= 0:
            continue
        path_times_tensor = path_times_dict[(origin, destination)]
        path_probs = F.softmax(-lambda_param * path_times_tensor, dim=0)
        for path, prob in zip(paths, path_probs):
            assigned_flow = demand * prob
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                new_flows[u, v] += assigned_flow
    with torch.no_grad():
        for u, v in G.edges:
            flow = new_flows[u, v]
            t0 = dist_matrix[u, v]
            G[u][v]['distance'] = bpr_function(t0, flow, capacity)
    return new_flows


def run_dta_logit_fast(od_matrix, G, dist_matrix, lambda_param, max_iter, tol, capacity, od_paths, path_times_dict):
    N = od_matrix.shape[0]
    flow_results = torch.zeros((N, N), dtype=torch.float32)
    for _ in range(max_iter):
        flow_matrix = logit_traffic_assignment_fast(od_matrix, od_paths, path_times_dict, G, dist_matrix, capacity, lambda_param)
        flow_results += flow_matrix
    return flow_results


class LeastSquaresOptimizer:
    def __init__(self, N, init_matrix, G, dist_matrix, lambda_param, max_iter, tol, capacity, od_paths, path_times_dict):
        self.N = N
        self.ode_shape = (N, N)
        self.init_ode = init_matrix.flatten()
        self.G = G
        self.dist_matrix = dist_matrix
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.capacity = capacity
        self.od_paths = od_paths
        self.path_times_dict = path_times_dict

    def forward_model(self, ode_flat):
        ode_matrix = ode_flat.reshape(self.ode_shape)
        flow_pred = run_dta_logit_fast(torch.tensor(ode_matrix, dtype=torch.float32),
                                       self.G,
                                       self.dist_matrix,
                                       self.lambda_param,
                                       self.max_iter,
                                       self.tol,
                                       self.capacity,
                                       self.od_paths,
                                       self.path_times_dict)
        return flow_pred.numpy().flatten()

    def optimize(self, observed_flow, max_nfev=2):
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
                               ftol=1e-2,
                               xtol=1e-2,
                               gtol=1e-2,
                               x_scale='jac',
                               tr_solver='lsmr',
                               verbose=2)

        return result.x.reshape(self.ode_shape)


import concurrent.futures

def process_sample(i):
    print(f"Processing sample {i + 1}/{num_samples}", flush=True)
    print(f"真实OD:{od[i].sum():.2f}", flush=True)
    print(f"初始ODE:{init_matrix.sum():.2f}", flush=True)

    local_G = G.copy()
    for u, v in local_G.edges:
        local_G[u][v]['distance'] = dist_matrix[u, v]

    optimizer = LeastSquaresOptimizer(N, init_matrix, local_G, dist_matrix,
                                      lambda_param=1, max_iter=1, tol=tol, capacity=500,
                                      od_paths=od_paths, path_times_dict=path_times_dict)

    optimized_ode = optimizer.optimize(flow[i])
    final_flow = optimizer.forward_model(optimized_ode.flatten()).reshape(N, N)

    loss = np.sum(((final_flow[flow[i] != 0] - flow[i][flow[i] != 0]) / flow[i][flow[i] != 0]) ** 2)

    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    pred_od_tensor = torch.tensor(optimized_ode, dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od_tensor, od_tensor)

    print(f"最终ODE:{optimized_ode.sum():.2f}", flush=True)
    print(f"真实OD:{od[i].sum():.2f}", flush=True)
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}", flush=True)

    return loss, rmse, mae, mape


# ========== MAIN ========== #
adj_matrix = np.load('../data/adj110_3.17.npy')
dist_matrix = np.load('../data/dist110_3.17.npy')

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


# 创建保存目录
output_dir = "results/predicted_od"
os.makedirs(output_dir, exist_ok=True)



start_time = time.time()

# 构建图并预计算路径
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
for i, j in G.edges:
    G[i][j]['distance'] = dist_matrix[i, j]
    G[i][j]['flow'] = 0
od_paths = precompute_paths(G, max_path_len=2)
path_times_dict = compute_path_times(od_paths, dist_matrix)

for i in range(num_samples):
    print("--------------------------",flush=True)
    print(f"Processing sample {i + 1}/{num_samples}",flush=True)
    print(f"真实OD:{od[i].sum():.2f}",flush=True)
    print(f"初始ODE:{init_matrix.sum():.2f}",flush=True)

    optimizer = LeastSquaresOptimizer(N, init_matrix, G, dist_matrix,
                                      lambda_param=1, max_iter=1, tol=tol, capacity=500,
                                      od_paths=od_paths, path_times_dict=path_times_dict)

    optimized_ode = optimizer.optimize(flow[i])
    final_flow = optimizer.forward_model(optimized_ode.flatten()).reshape(N, N)

    loss = np.sum(((final_flow[flow[i] != 0] - flow[i][flow[i] != 0]) / flow[i][flow[i] != 0]) ** 2)
    loss_total += loss

    od_tensor = torch.tensor(od[i], dtype=torch.float32)
    pred_od_tensor = torch.tensor(optimized_ode, dtype=torch.float32)
    rmse, mae, mape = calculate_rmse_mae(pred_od_tensor, od_tensor)

    print(f"最终ODE:{optimized_ode.sum():.2f}",flush=True)
    print(f"真实OD:{od[i].sum():.2f}",flush=True)
    print(f"样本{i}的RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}",flush=True)

    rmse_total += rmse
    mae_total += mae
    mape_total += mape

rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"Total Test loss: {loss_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}",flush=True)

end_time = time.time()
time_difference_seconds = end_time - start_time
times = time_difference_seconds / 60
print(f"耗时{times:.2f}min",flush=True)

log_filename = f"log/SSM_LeastSquares.log"
with open(log_filename, 'a') as log_file:
    log_file.write(
        f"Samples: {num_samples}, MaxIter: {num_epochs}, Tol: {tol}, Test Loss: {loss_test:.4f} RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}, Times: {times}\n",flush=True)
