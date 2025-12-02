import time

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge

def calculate_rmse_mae(predictions, targets):
    # 计算均方误差
    mse = np.mean((predictions - targets) ** 2)
    # 计算均方根误差
    rmse = np.sqrt(mse)
    # 计算平均绝对误差
    mae = np.mean(np.abs(predictions - targets))

    # 计算 MAPE，避免除以零
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(
            np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = 0.0

    return rmse, mae, mape


np.random.seed(42)


# ================================
# 仿真器函数（多链路支持）
# ================================
def simulator(d,A):
    """
    多链路模拟器
      od对 d = [N,]
      分配矩阵 A = [L,N]
      :return Pd = [L,]
    """
    return A @ d


# ================================
# 初始采样
# ================================
def initial_sampling(n_samples, bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))


# ================================
# 贝叶斯回归估计模型 M2： P, phi_hat
# ================================
def estimate_M2_bayesian_multi(X_samples, V_samples):
    model = MultiOutputRegressor(BayesianRidge())
    model.fit(X_samples, V_samples)
    P_est = np.vstack([est.coef_ for est in model.estimators_])  # [L, N]
    phi_hat = np.array([
        np.sqrt(1.0 / est.alpha_ + 1.0 / est.lambda_) for est in model.estimators_
    ])
    print(f"P:{P_est.shape},phi:{phi_hat.shape}")
    return P_est, phi_hat


# ================================
# 物理约束函数
# ================================
def physical_model_constraints(d, v_f, P, r, phi_hat):
    Pd = np.dot(P, d)
    lower = v_f - r * phi_hat
    upper = v_f + r * phi_hat
    return np.all((Pd >= lower) & (Pd <= upper))


# ================================
# Acquisition Function
# ================================
def acquisition_function(d, gp, P, v_f, r, phi_hat, gamma=1.0, rho=5.0):
    d = d.reshape(1, -1)
    mean, std = gp.predict(d, return_std=True)
    mean, std = mean[0], std[0]
    Pd = np.dot(P, d.flatten())
    dist_lower = np.min(Pd - (v_f - r * phi_hat))
    dist_upper = np.min((v_f + r * phi_hat) - Pd)
    h_d = np.minimum(dist_lower, dist_upper)
    pwf = - (h_d ** 2) / (rho ** 2) + 2 * h_d / rho if h_d <= rho else 1.0
    return mean - gamma * pwf * std


# ================================
# 选取下一个采样点
# ================================
def propose_location(gp, bounds, P, v_f, r, phi_hat):
    phi_hat = np.maximum(phi_hat, 300.0)
    # print(phi_hat)

    dim = bounds.shape[0]
    best_x, best_acq = None, np.inf
    for i in range(2):
        print(f"采样次数:{i}")
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        constraints = [
            {'type': 'ineq', 'fun': lambda d: np.min(np.dot(P, d) - (v_f - r * phi_hat))},
            {'type': 'ineq', 'fun': lambda d: np.min((v_f + r * phi_hat) - np.dot(P, d))}
        ]
        # constraints = [
        #     {'type': 'ineq', 'fun': lambda d: np.dot(P, d) - (v_f - r * phi_hat)},
        #     {'type': 'ineq', 'fun': lambda d: (v_f + r * phi_hat) - np.dot(P, d)}
        # ]
        res = minimize(lambda d: acquisition_function(d, gp, P, v_f, r, phi_hat),
                       x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')
        if res.success and res.fun < best_acq:
            best_acq = res.fun
            best_x = res.x
    return best_x

def get_bounds(true_od):
    # 计算每个 OD 对在所有时间步上的最大值和最小值
    T, N, _ = true_od.shape
    # 将数据从 [T,N,N] 重塑为 [T,N*N]
    reshaped_od = true_od.reshape(T, N * N)
    # 计算每列的最大值和最小值
    max_values = np.max(reshaped_od, axis=0)
    min_values = np.min(reshaped_od, axis=0)
    # 组合最大最小值为 [N*N, 2] 的矩阵
    bound = np.column_stack((min_values, max_values))
    return bound
# ================================
# 主流程
# ================================
def dynamic_OD_estimation(n_initial=5, max_iter=3):
    # bounds = np.array([[0, 100], [0, 100]])  # OD变量范围
    # 定义超参数 N
    N = 110*110
    N_sub = 110*110


    A = np.load("data/贝叶斯估计的系数矩阵P_LN.npy")
    A = A[:,:N_sub]

    # v_f = np.array([100.0, 90.0])  # 两条链路的真实观测流量 L=2
    r = 1

    num_samples = 1

    true_od = np.load("data/OD_完整批处理_3.17_Final.npy")  # [T,N,N]

    bounds = get_bounds(true_od) # [N*N,2]
    bounds = bounds[:N_sub,:]

    T = true_od.shape[0]
    true_od = true_od.reshape(T,-1) # [T, N*N]
    true_od = true_od[-num_samples:].astype(np.float32)
    true_od = true_od[:,:N_sub]

    v_f = np.load("data/Link_flow_TL_3.19_可微.npy")  # [T,L]
    v_f = v_f[-num_samples:].astype(np.float32)

    rmse_total = 0
    mae_total = 0
    mape_total = 0

    start_time = time.time()

    for i in range(num_samples):
        print(f"------------样本{i}---------------")
        # Step 1: 初始采样
        X = initial_sampling(n_initial, bounds)  # shape = (n, N)
        print(f"OD:{X.shape}")
        V = np.array([simulator(d, A) for d in X])  # shape = (n, L)
        print(f"Vs:{V.shape}")
        # Step 2: 贝叶斯回归估计 M2 的 P, phi_hat
        P, phi_hat = estimate_M2_bayesian_multi(X, V)

        # Step 3: 使用仿真误差作为目标函数
        y = []
        for d in X:
            non_zero_mask = v_f[i] != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs(simulator(d, A) - v_f[i])[non_zero_mask] / v_f[i][non_zero_mask])
            else:
                mape = 0.0
            y.append(mape)
        y = np.array(y)

        kernel = C(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        for iteration in range(max_iter):
            gp.fit(X, y)
            d_new = propose_location(gp, bounds, P, v_f[i], r, phi_hat)
            if d_new is None:
                print(f"迭代 {iteration} 找不到可行点，提前终止")
                break
            # 修改此处，避免除以零
            non_zero_mask = v_f[i] != 0
            if non_zero_mask.sum() > 0:
                f_new = np.mean(np.abs(simulator(d_new, A) - v_f[i])[non_zero_mask] / v_f[i][non_zero_mask])
            else:
                f_new = 0.0
            X = np.vstack((X, d_new))
            y = np.append(y, f_new)

            # 更新 M2 估计
            V = np.array([simulator(d, A) for d in X])
            P, phi_hat = estimate_M2_bayesian_multi(X, V)

            print(f"Iteration {iteration + 1}: d = {d_new.shape}, MAPE = {f_new:.4f}")

        best_idx = np.argmin(y)
        d_opt = X[best_idx]
        f_opt = y[best_idx]

        print("\n最终最优解:")
        print(f"OD向量 d* {d_opt.shape}")
        print(f"最小交通流量观测误差MAPE = {f_opt:.4f}")

        rmse, mae, mape = calculate_rmse_mae(d_opt, true_od[i])
        rmse_total += rmse
        mae_total += mae
        mape_total += mape

    # 计算平均评估指标
    rmse_test = rmse_total / num_samples
    mae_test = mae_total / num_samples
    mape_test = mape_total / num_samples

    print(f"Total Test RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}")

    end_time = time.time()
    time_difference_seconds = end_time - start_time
    times = time_difference_seconds / 60
    print(f"耗时{times:.2f}min")

    log_filename = f"log/贝叶斯估计.log"
    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Samples: {num_samples}, Total RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f}, Times: {times}\n")


if __name__ == "__main__":
    dynamic_OD_estimation()
