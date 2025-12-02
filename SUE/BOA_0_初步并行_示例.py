import time
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import BayesianRidge
from joblib import Parallel, delayed
import os
import logging

import functools
print = functools.partial(print, flush=True)


def calculate_rmse_mae(predictions, targets):
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    non_zero_mask = targets != 0
    mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])) if non_zero_mask.sum() > 0 else 0.0
    return rmse, mae, mape


def simulator(d, A):
    return A @ d


def initial_sampling(n_samples, bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))


def estimate_M2_bayesian_multi_parallel(X_samples, V_samples):
    '''
    建立样本的OD对向单个路段flowi的贝叶斯回归模型
    :param X_samples:
    :param V_samples:
    :return:
    '''
    def fit_single_output(y):
        model = BayesianRidge()
        model.fit(X_samples, y)
        return model.coef_, np.sqrt(1.0 / model.alpha_ + 1.0 / model.lambda_)

    results = Parallel(n_jobs=-1)(delayed(fit_single_output)(V_samples[:, i]) for i in range(V_samples.shape[1]))
    P_est = np.vstack([r[0] for r in results])
    phi_hat = np.array([r[1] for r in results])
    print(f"P:{P_est.shape},phi:{phi_hat.shape}",flush=True)
    return P_est, phi_hat


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


def propose_location(gp, bounds, P, v_f, r, phi_hat):
    phi_hat = np.maximum(phi_hat, 10.0)
    dim = bounds.shape[0]
    best_x, best_acq = None, np.inf
    for i in range(2):
        print(f"采样次数:{i}",flush=True)
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        constraints = [
            {'type': 'ineq', 'fun': lambda d: np.min(np.dot(P, d) - (v_f - r * phi_hat))},
            {'type': 'ineq', 'fun': lambda d: np.min((v_f + r * phi_hat) - np.dot(P, d))}
        ]
        # 用优化算法找到使分数最小的点
        res = minimize(lambda d: acquisition_function(d, gp, P, v_f, r, phi_hat),
                       x0=x0, bounds=bounds, constraints=constraints, method='SLSQP'
                       ,options={'maxiter': 100, 'ftol': 1e-3})

        if res.success and res.fun < best_acq:
            best_acq = res.fun
            best_x = res.x
    return best_x





def process_sample(i, A, bounds, true_od, v_f, n_initial, max_iter, r):
    print(f"进入样本{i}的初始点采样",flush=True)
    X = initial_sampling(n_initial, bounds)
    V = np.array([simulator(d, A) for d in X])
    P, phi_hat = estimate_M2_bayesian_multi_parallel(X, V)

    y = []
    for d in X:
        non_zero_mask = v_f[i] != 0
        mape = np.mean(np.abs(simulator(d, A) - v_f[i])[non_zero_mask] / v_f[i][non_zero_mask]) if non_zero_mask.sum() > 0 else 0.0
        y.append(mape)
    y = np.array(y)

    # 原写法
    # gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-6, normalize_y=True)

    ## 健壮性
    kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5  # 增加优化器重启次数
    )
    for iteration in range(max_iter):
        print(f"样本{i}的第{iteration}次迭代",flush=True)
        gp.fit(X, y)
        d_new = propose_location(gp, bounds, P, v_f[i], r, phi_hat)
        if d_new is None:
            print(f"样本{i}的第{iteration}次迭代采样失败")
            break
        non_zero_mask = v_f[i] != 0
        f_new = np.mean(np.abs(simulator(d_new, A) - v_f[i])[non_zero_mask] / v_f[i][non_zero_mask]) if non_zero_mask.sum() > 0 else 0.0
        X = np.vstack((X, d_new))
        y = np.append(y, f_new)
        V = np.array([simulator(d, A) for d in X])
        P, phi_hat = estimate_M2_bayesian_multi_parallel(X, V)

    best_idx = np.argmin(y)
    d_opt = X[best_idx]
    f_opt = y[best_idx]
    print(f"样本{i} Epoch：{iteration}的OD估计值和OD真值:{d_opt} \n{true_od[i]}")
    rmse, mae, mape = calculate_rmse_mae(d_opt, true_od[i])
    print(f"样本{i} Epoch：{iteration} RMSE:{rmse:.4f} MAE:{mae:.4f} MAPE:{mape:.4f}")
    return rmse, mae, mape


def dynamic_OD_estimation(n_initial=4, max_iter=50):
    N = 6

    true_od = np.array([
        [10,10,10,10,10,10],
        [20,20,20,20,20,20],
        [30,30,30,30,30,30],
    ])



    A = np.array([
        [0.66, 0, 0, 0.33, 0.33, 0],
        [0.33, 0, 0, 0.66, 0, 0.33],
        [0, 0.33, 0, 0.33, 0, 0.66],
        [0, 0.66, 0.33, 0, 0, 0.33],
        [0, 0.33, 0.66, 0, 0.33, 0],
        [0.33, 0, 0.33, 0, 0.66, 0]

    ])

    # 模拟观测的 link flow：V = A @ d
    v_f = np.array([A @ od for od in true_od], dtype=np.float32)

    # 构造 bounds，使用 min-max 范围
    bounds = np.column_stack((np.min(true_od, axis=0), np.max(true_od, axis=0)))

    print(f"分配矩阵的shape:{A.shape}")
    print(f"真实OD的shape：:{true_od.shape}")
    print(f"观测路段流量的shape：:{v_f.shape}")

    num_samples = 3

    r = 1

    start_time = time.time()

    results = Parallel(n_jobs=-1)(
        delayed(process_sample)(i, A, bounds, true_od, v_f, n_initial, max_iter, r)
        for i in range(num_samples)
    )

    rmse_total = sum(r for r, _, _ in results)
    mae_total = sum(m for _, m, _ in results)
    mape_total = sum(mp for _, _, mp in results)

    print(f"Total Test RMSE: {rmse_total / num_samples:.4f}, MAE: {mae_total / num_samples:.4f}, MAPE: {mape_total / num_samples:.4f}",flush=True)

    time_taken = (time.time() - start_time) / 60
    print(f"耗时 {time_taken:.2f} min",flush=True)

    os.makedirs("log", exist_ok=True)
    with open("log/贝叶斯估计_示例.log", 'a') as log_file:
        log_file.write(f"Samples: {num_samples}, Total RMSE: {rmse_total / num_samples:.4f} MAE: {mae_total / num_samples:.4f} MAPE: {mape_total / num_samples:.4f}, Time: {time_taken:.2f} min\n")


if __name__ == "__main__":
    dynamic_OD_estimation()
