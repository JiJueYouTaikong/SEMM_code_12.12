import time
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import BayesianRidge
from joblib import Parallel, delayed
import os
import logging

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
    def fit_single_output(y):
        model = BayesianRidge()
        model.fit(X_samples, y)
        return model.coef_, np.sqrt(1.0 / model.alpha_ + 1.0 / model.lambda_)

    results = Parallel(n_jobs=-1)(delayed(fit_single_output)(V_samples[:, i]) for i in range(V_samples.shape[1]))
    P_est = np.vstack([r[0] for r in results])
    phi_hat = np.array([r[1] for r in results])
    print(f"P:{P_est.shape},phi:{phi_hat.shape}")
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
    phi_hat = np.maximum(phi_hat, 300.0)
    dim = bounds.shape[0]
    best_x, best_acq = None, np.inf
    for i in range(2):
        print(f"采样次数:{i}")
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        constraints = [
            {'type': 'ineq', 'fun': lambda d: np.min(np.dot(P, d) - (v_f - r * phi_hat))},
            {'type': 'ineq', 'fun': lambda d: np.min((v_f + r * phi_hat) - np.dot(P, d))}
        ]
        res = minimize(lambda d: acquisition_function(d, gp, P, v_f, r, phi_hat),
                       x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')
        if res.success and res.fun < best_acq:
            best_acq = res.fun
            best_x = res.x
    return best_x


def get_bounds(true_od):
    T, N, _ = true_od.shape
    reshaped_od = true_od.reshape(T, N * N)
    max_values = np.max(reshaped_od, axis=0)
    min_values = np.min(reshaped_od, axis=0)
    return np.column_stack((min_values, max_values))


def process_sample(i, A, bounds, true_od, v_f, n_initial, max_iter, r):
    print(f"进入样本{i}的初始点采样")
    X = initial_sampling(n_initial, bounds)
    V = np.array([simulator(d, A) for d in X])
    P, phi_hat = estimate_M2_bayesian_multi_parallel(X, V)

    y = []
    for d in X:
        non_zero_mask = v_f[i] != 0
        mape = np.mean(np.abs(simulator(d, A) - v_f[i])[non_zero_mask] / v_f[i][non_zero_mask]) if non_zero_mask.sum() > 0 else 0.0
        y.append(mape)
    y = np.array(y)

    gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-6, normalize_y=True)

    for iteration in range(max_iter):
        print(f"样本{i}的第{iteration}次迭代")
        gp.fit(X, y)
        d_new = propose_location(gp, bounds, P, v_f[i], r, phi_hat)
        if d_new is None:
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
    rmse, mae, mape = calculate_rmse_mae(d_opt, true_od[i])
    return rmse, mae, mape


def dynamic_OD_estimation(n_initial=5, max_iter=3):
    N = 110*110

    A = np.load("data/贝叶斯估计的系数矩阵P_LN.npy")[:, :N]

    true_od = np.load("data/OD_完整批处理_3.17_Final.npy")

    bounds = get_bounds(true_od)[:N, :]

    T = true_od.shape[0]
    true_od = true_od.reshape(T, -1).astype(np.float32)[:, :N]

    v_f = np.load("data/Link_flow_TL_3.19_可微.npy").astype(np.float32)

    num_samples = 3  # 可以改为并行多个样本

    true_od = true_od[-num_samples:]
    v_f = v_f[-num_samples:]

    r = 1

    start_time = time.time()

    results = Parallel(n_jobs=-1)(
        delayed(process_sample)(i, A, bounds, true_od, v_f, n_initial, max_iter, r)
        for i in range(num_samples)
    )

    rmse_total = sum(r for r, _, _ in results)
    mae_total = sum(m for _, m, _ in results)
    mape_total = sum(mp for _, _, mp in results)

    print(f"Total Test RMSE: {rmse_total / num_samples:.4f}, MAE: {mae_total / num_samples:.4f}, MAPE: {mape_total / num_samples:.4f}")

    time_taken = (time.time() - start_time) / 60
    print(f"耗时 {time_taken:.2f} min")

    os.makedirs("log", exist_ok=True)
    with open("log/贝叶斯估计.log", 'a') as log_file:
        log_file.write(f"Samples: {num_samples}, Total RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f}, Time: {time_taken:.2f} min\n")


if __name__ == "__main__":
    dynamic_OD_estimation()
