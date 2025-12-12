def calculate_rmse_mae(predictions, targets):
    # （保持原实现不变）
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    non_zero_mask = targets != 0
    mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[
        non_zero_mask]) if non_zero_mask.sum() > 0 else 0.0
    return rmse, mae, mape


# ================================
# 优化点1：向量化仿真器 + 稀疏矩阵支持
# ================================
def simulator(d, A):
    """向量化仿真器"""
    if isinstance(d, np.ndarray) and d.ndim == 2:
        return d @ A.T  # 批量计算
    else:
        return A @ d  # 单样本计算


# ================================
# 优化点2：并行化贝叶斯回归
# ================================
def estimate_M2_bayesian_multi(X_samples, V_samples):
    model = MultiOutputRegressor(BayesianRidge(), n_jobs=-1)  # 启用并行
    model.fit(X_samples, V_samples)
    P_est = np.vstack([est.coef_ for est in model.estimators_])
    phi_hat = np.array([np.sqrt(1.0 / est.alpha_ + 1.0 / est.lambda_) for est in model.estimators_])
    print(f"P:{P_est.shape}, phi:{phi_hat.shape}")
    return P_est, phi_hat


# ================================
# 优化点3：候选点采样策略代替数值优化
# ================================
def propose_location(gp, bounds, P, v_f, r, phi_hat, n_candidates=5000):
    """
    新策略：生成大量候选点后筛选
    返回形状 (n_dim, )
    """
    # 生成候选点（并行化）
    candidates = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_candidates, bounds.shape[0]))

    # 向量化约束检查
    Pd = candidates @ P.T  # 形状 (n_candidates, L)
    lower = v_f - r * phi_hat
    upper = v_f + r * phi_hat
    valid_mask = np.all((Pd >= lower) & (Pd <= upper), axis=1)

    if valid_mask.sum() == 0:
        print("警告：无有效候选点")
        return None

    valid_candidates = candidates[valid_mask]

    # 并行计算获取函数
    acq_values = Parallel(n_jobs=-1)(
        delayed(acquisition_function)(cand, gp, P, v_f, r, phi_hat)
        for cand in valid_candidates
    )

    return valid_candidates[np.argmin(acq_values)]


# ================================
# 优化点4：预计算加速获取函数
# ================================
def acquisition_function(d, gp, P, v_f, r, phi_hat, gamma=1.0, rho=5.0):
    d = d.reshape(1, -1)
    mean, std = gp.predict(d, return_std=True)
    mean, std = mean[0], std[0]

    # 预计算边界距离
    Pd = d @ P.T  # 形状 (1, L)
    dist_lower = Pd - (v_f - r * phi_hat)
    dist_upper = (v_f + r * phi_hat) - Pd
    h_d = np.min(np.hstack([dist_lower, dist_upper]), axis=1)

    # 分段函数优化
    h_d = h_d[0]  # 因为输入是单样本
    pwf = - (h_d ** 2) / (rho ** 2) + 2 * h_d / rho if h_d <= rho else 1.0
    return mean - gamma * pwf * std


# ================================
# 主流程优化
# ================================
def dynamic_OD_estimation(n_initial=5, max_iter=3):
    # 加载数据时转换为稀疏矩阵
    A = csr_matrix(np.load("data/贝叶斯估计的系数矩阵P_LN.npy")[:, :N_sub])

    # 预加载数据
    true_od = np.load("data/OD_完整批处理_3.17_Final.npy").reshape(-1, N * N)[-num_samples:, :N_sub]
    v_f = np.load("data/Link_flow_TL_3.19_可微.npy")[-num_samples:].astype(np.float32)

    # 初始化存储
    metrics = {'rmse': [], 'mae': [], 'mape': []}

    for i in range(num_samples):
        # 优化点5：向量化初始采样计算
        X = initial_sampling(n_initial, bounds)
        V = simulator(X, A)  # 直接矩阵乘法

        # 优化点6：并行计算初始MAPE
        with Parallel(n_jobs=-1) as parallel:
            y = parallel(
                delayed(calculate_mape)(simulator(d, A), v_f[i])
                for d in X
            )
        y = np.array(y)

        # 优化高斯过程配置
        gp = GaussianProcessRegressor(
            kernel=C(1.0) * RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2  # 减少重启次数
        )

        for iteration in range(max_iter):
            gp.fit(X, y)
            d_new = propose_location(gp, bounds, P, v_f[i], r, phi_hat)

            if d_new is None:
                print(f"迭代 {iteration} 无有效点")
                break

            # 批量更新
            X = np.vstack((X, d_new))
            y = np.append(y, calculate_mape(simulator(d_new, A), v_f[i]))

            # 增量式更新估计模型（替代完全重新计算）
            V_new = simulator(d_new, A)
            P, phi_hat = update_M2_estimator(X, V, P, phi_hat)  # 需实现增量更新