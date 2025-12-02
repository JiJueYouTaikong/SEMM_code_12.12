import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from sub_utils.计算PESUELOGIT需要的M和D import init_M_D_matrices, create_graph
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

    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(
            torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()


def bpr_t(t_min, x_max, x_hat_i, alpha, beta):
    term = (1 + alpha * (x_hat_i / x_max)) ** beta
    t_i = t_min * term
    return t_i


class TrafficModel(nn.Module):
    def __init__(self, num_links, num_paths, num_od_pairs, D, M, q_his, t_min, x_max):
        super(TrafficModel, self).__init__()
        self.x_hat = nn.Parameter(torch.zeros(num_links))
        self.alpha = nn.Parameter(torch.full((num_links,), 0.15))
        self.beta = nn.Parameter(torch.full((num_links,), 4.0))

        mean_q_his = torch.mean(q_his)
        noise = torch.normal(mean=torch.zeros_like(q_his), std=0.1 * mean_q_his)
        self.q_hat = nn.Parameter(q_his + noise).float()

        self.D = torch.tensor(D, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

        # ✅ 新增：把 t_min 和 x_max 保存为成员变量
        self.t_min = t_min
        self.x_max = x_max

    def forward(self):
        # ✅ 使用 self.t_min, self.x_max
        t = bpr_t(self.t_min, self.x_max, self.x_hat, self.alpha, self.beta)
        path_utilities = torch.matmul(self.D.T, t)
        path_selection_probabilities = torch.zeros(self.M.shape[1])
        for i in range(self.M.shape[0]):
            path_indices = torch.where(self.M[i] == 1)[0]
            if len(path_indices) > 0:
                od_path_utilities = path_utilities[path_indices]
                p = F.softmax(od_path_utilities, dim=0)
                path_selection_probabilities[path_indices] = p

        p = path_selection_probabilities
        q_vector = torch.matmul(self.M.T, self.q_hat)
        f = q_vector * p
        x = torch.matmul(f, self.D.T)
        return x, t, f, p


def convergence_check(x, x_hat, epsilon=1e-5):
    numerator = torch.linalg.norm(x_hat - x, ord=1)
    denominator = torch.linalg.norm(x, ord=1)
    rho_j = numerator / denominator if denominator != 0 else 0
    return rho_j < epsilon


def loss_function(x, x_hat, x_obs, q_hat, q_his):
    loss_x = torch.linalg.norm(x_hat - x_obs) ** 2
    loss_e = torch.linalg.norm(x - x_hat) ** 2
    loss_q = torch.linalg.norm(q_hat - q_his) ** 2
    loss = loss_x + loss_e + loss_q
    return loss


def train_single_sample(i, D, M, q_his, x_obs, q_obs, t_min, x_max, max_iter, lrate, return_dict):
    model = TrafficModel(D.shape[0], D.shape[1], M.shape[0], D, M, q_his, t_min, x_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    for j in range(max_iter):
        x_hat = model.x_hat.detach().clone()
        x, t, f, p = model()
        if convergence_check(x, x_hat):
            print(f"[Sample {i}] Converged at iteration {j}")
            break
        loss = loss_function(x, model.x_hat, x_obs[i], model.q_hat, q_his)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 20 == 0 or j == max_iter - 1:
            print(f"[Sample {i}] Epoch {j}, Loss: {loss.item():.4f}")

    print(f"[Sample {i}] True OD Sum: {int(sum(q_obs[i]))}")
    print(f"[Sample {i}] Estimated OD Sum: {int(sum(model.q_hat))}")

    rmse, mae, mape = calculate_rmse_mae(q_obs[i], model.q_hat)
    print(f"[Sample {i}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")

    return_dict[i] = (rmse, mae, mape)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # --- 加载数据 ---
    adj = np.load("../data/adj110_3.17.npy")
    dist = np.load("../data/dist110_3.17.npy")
    OD = np.load("../data/OD_完整批处理_3.17_Final.npy")
    OD = OD[0]
    G = create_graph(adj, dist)
    M, D = init_M_D_matrices(OD, adj, G)

    print(f"M:{M.shape}, D:{D.shape}")

    num_links = D.shape[0]
    num_paths = D.shape[1]
    num_od_pairs = M.shape[0]

    t_min = torch.full((num_links,), 2)
    x_max = torch.full((num_links,), 500)

    log_filename = f"log/PESUELOGIT.log"

    x_obs = np.load("data/Link_flow_TL_MSA-SUE_logit3_6_14.npy")
    q_obs = np.load("data/OD_完整批处理_3.17_Final.npy")
    T = q_obs.shape[0]
    print(f"数据集的总时间步:{T}")
    num_samples = 35

    q_his = q_obs[:T - num_samples]
    q_his = np.mean(q_his, axis=0)
    q_his = torch.from_numpy(flatten_his_od(q_his))

    q_obs = q_obs[-num_samples:]
    q_obs = flatten_od(q_obs)
    q_obs = torch.from_numpy(q_obs)

    x_obs = x_obs[-num_samples:]
    x_obs = torch.from_numpy(x_obs)

    print(f"历史OD对:{q_his.shape}, 时变OD对观测值:{q_obs.shape}, 时变链路流量观测值:{x_obs.shape}")

    # --- 并行训练 ---
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    max_iter = 2000
    lrate = 0.1

    for i in range(num_samples):
        p = mp.Process(
            target=train_single_sample,
            args=(i, D, M, q_his, x_obs, q_obs, t_min, x_max, max_iter, lrate, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_rmse = total_mae = total_mape = 0
    for i in range(num_samples):
        rmse, mae, mape = return_dict[i]
        total_rmse += rmse
        total_mae += mae
        total_mape += mape

    total_rmse /= num_samples
    total_mae /= num_samples
    total_mape /= num_samples

    print(f"[Final] Average RMSE: {total_rmse:.4f}, MAE: {total_mae:.4f}, MAPE: {total_mape:.4f}")
    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Lr={lrate}, Iter={max_iter}, Samples={num_samples} Final RMSE: {total_rmse:.4f} MAE: {total_mae:.4f} MAPE: {total_mape:.4f}\n")
