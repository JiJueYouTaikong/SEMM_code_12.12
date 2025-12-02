import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 设置随机数种子，保证结果可复现
np.random.seed(42)


# ================================
# Step 0: 仿真器函数
# ================================
def simulator(d):
    """
    给定OD向量d，输出模拟的交通流。稀疏矩阵P和协方差矩阵由贝叶斯优化
    参数:
        d (np.ndarray): OD向量，形状为 (N,)，N 是OD向量的维度
    返回:
        float: 模拟的交通流值
    """
    noise = np.random.normal(0, 2)
    return 0.8 * d[0] + 1.2 * d[1] + noise


# ================================
# Step 1: 初始化样本
# ================================
def initial_sampling(n_samples, bounds):
    """
    在给定边界内随机生成初始样本
    参数:
        n_samples (int): 要生成的样本数量
        bounds (np.ndarray): 样本每个维度的取值范围，形状为 (N, 2)，N 是OD向量的维度
    返回:
        np.ndarray: 生成的初始样本，形状为 (n_samples, N)
    """
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))


# ================================
# Step 2: 构建物理代理模型 (M2)
# ================================
def physical_model_constraints(d, v_f, P, r, phi_hat):
    """
    物理代理模型约束：vf - r*phi <= Pd <= vf + r*phi
    参数:
        d (np.ndarray): OD向量，形状为 (N,)，N 是OD向量的维度
        v_f (np.ndarray): 真实交通流观测，形状为 (L,)，L 是观测的数量
        P (np.ndarray): 系数矩阵，形状为 (L, N)
        r (float): 物理约束的放松系数。
        phi_hat (np.ndarray): 交通量波动范围，形状为 (L,)
    返回:
        bool: 如果满足约束条件返回 True，否则返回 False
    """
    # 计算 Pd
    Pd = np.dot(P, d)
    # 计算约束的下界
    lower = v_f - r * phi_hat
    # 计算约束的上界
    upper = v_f + r * phi_hat
    # 检查 Pd 是否在上下界之间
    return np.all((Pd >= lower) & (Pd <= upper))


# ================================
# Step 3: 采样新点（基于 GP + PWF）
# ================================
def acquisition_function(d, gp, P, v_f, r, phi_hat, gamma=1.0, rho=5.0):
    """
    定义Acquisition Function，考虑物理约束和PWF调整
    参数:
        d (np.ndarray): OD向量，形状为 (N,)，N 是OD向量的维度
        gp (GaussianProcessRegressor): 高斯过程回归模型
        P (np.ndarray): 系数矩阵，形状为 (L, N)
        v_f (np.ndarray): 真实交通流观测，形状为 (L,)，L 是观测的数量
        r (float): 物理约束的放松系数
        phi_hat (np.ndarray): 交通量波动范围，形状为 (L,)
        gamma (float): 平衡均值和标准差的系数，默认为 1.0
        rho (float): PWF 函数的参数，默认为 5.0
    返回:
        float: Acquisition Function 的值
    """
    # 将 d 转换为二维数组
    d = d.reshape(1, -1)
    # 使用高斯过程模型预测均值和标准差
    mean, std = gp.predict(d, return_std=True)
    mean = mean[0]
    std = std[0]

    # Projection Distance Function h(d)
    # 计算 Pd
    Pd = np.dot(P, d.flatten())
    # 计算 Pd 到下界的最小距离
    dist_lower = np.min(Pd - (v_f - r * phi_hat))
    # 计算 Pd 到上界的最小距离
    dist_upper = np.min((v_f + r * phi_hat) - Pd)
    # 取两者中的最小值
    h_d = np.minimum(dist_lower, dist_upper)

    # PWF权重
    if h_d <= rho:
        # 当 h_d 小于等于 rho 时，计算 PWF 权重
        pwf = - (h_d ** 2) / (rho ** 2) + 2 * h_d / rho
    else:
        # 当 h_d 大于 rho 时，PWF 权重为 1
        pwf = 1.0

    # 计算 Acquisition Function 的值
    acq = mean - gamma * pwf * std
    return acq


def propose_location(gp, bounds, P, v_f, r, phi_hat):
    """
    优化Acquisition Function以选择下一个采样点
    参数:
        gp (GaussianProcessRegressor): 高斯过程回归模型
        bounds (np.ndarray): 样本每个维度的取值范围，形状为 (N, 2)，N 是OD向量的维度
        P (np.ndarray): 系数矩阵，形状为 (L, N)
        v_f (np.ndarray): 真实交通流观测，形状为 (L,)，L 是观测的数量
        r (float): 物理约束的放松系数
        phi_hat (np.ndarray): 交通量波动范围，形状为 (L,)
    返回:
        np.ndarray: 下一个采样点，形状为 (N,)
    """
    # 获取 OD 向量的维度
    dim = bounds.shape[0]

    best_x = None
    best_acq = np.inf

    # 多次随机启动，避免局部最优
    for _ in range(20):
        # 随机生成初始点
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)

        # 定义约束条件
        constraints = ({'type': 'ineq',
                        'fun': lambda d: np.min(np.dot(P, d) - (v_f - r * phi_hat))},
                       {'type': 'ineq',
                        'fun': lambda d: np.min((v_f + r * phi_hat) - np.dot(P, d))})

        # 使用 SLSQP 方法优化 Acquisition Function
        res = minimize(lambda d: acquisition_function(d, gp, P, v_f, r, phi_hat),
                       x0=x0,
                       bounds=bounds,
                       constraints=constraints,
                       method='SLSQP')

        # 如果优化成功且当前结果优于之前的最优结果
        if res.success and res.fun < best_acq:
            best_acq = res.fun
            best_x = res.x

    return best_x


# ================================
# Step 4: 主循环
# ================================
def dynamic_OD_estimation(n_initial=5, max_iter=20):
    """主流程"""

    # 设置问题参数 可行域边界
    bounds = np.array([[0, 100], [0, 100]])  # OD每个元素在0到100之间

    v_f = np.array([100])  # 真实交通流观测 true flow [L,]
    P = np.array([[0.8, 1.2]])  # 系数矩阵 [L,N]
    r = 1.0  # 物理约束的放松系数
    phi_hat = np.array([5])  # 交通量波动范围

    # 初始化采样
    X = initial_sampling(n_initial, bounds)  # [采样数量,N]
    # 仿真器对单个样本：Pd = [L,N]*[N,] = [L,]  --> 总：[采样数量,N]
    y = np.array([abs(simulator(d) - v_f[0]) / v_f[0] for d in X])  # [采样数量,] 计算MAPE

    # 高斯过程模型
    kernel = C(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    for iteration in range(max_iter):
        # 训练GP，基于[di,fi]对
        gp.fit(X, y)

        # 选取新采样点
        d_new = propose_location(gp, bounds, P, v_f, r, phi_hat)

        if d_new is None:
            print(f"在第{iteration}轮采样时无法找到可行点，提前终止")
            break

        # 仿真评估
        f_new = abs(simulator(d_new) - v_f[0]) / v_f[0]

        # 更新数据集
        X = np.vstack((X, d_new))
        y = np.append(y, f_new)

        print(f"Iteration {iteration + 1}: d = {d_new}, f(d) = {f_new:.4f}")

    # 用收集了初始点和新选点的点集训练最终GP，并寻找最优点
    gp.fit(X, y)
    best_idx = np.argmin(y)
    d_opt = X[best_idx]
    f_opt = y[best_idx]

    print("\n最终最优解:")
    print(f"OD matrix d* = {d_opt}")
    print(f"对应的MAPE = {f_opt:.4f}")

    return d_opt, f_opt


# ================================
# 运行主程序
# ================================
if __name__ == "__main__":
    dynamic_OD_estimation()
