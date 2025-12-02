import numpy as np


# 步骤2：求解动态用户均衡（DUE）
def solve_due(E_D, I, T, K_list):
    P_blocks = []
    for i in range(I):
        K_i = K_list[i]
        P_i_blocks = []
        for t in range(T):
            # 先随机生成选择比例矩阵
            P_i_t = np.random.rand(K_i, I * T)
            P_i_blocks.append(P_i_t)
        P_i = np.vstack(P_i_blocks)
        P_blocks.append(P_i)
    P = np.vstack(P_blocks)
    return P


# 步骤3：计算均值和方差-协方差矩阵 √√√
def calculate_mean_covariance(E_D, Sigma_D, P):
    E_F = np.dot(P, E_D)
    Sigma_DF = np.dot(Sigma_D, P.T)
    Sigma_F = np.dot(P, Sigma_DF)
    Sigma_FD = np.dot(P,Sigma_D)
    Sigma = np.block([[Sigma_D, Sigma_DF], [Sigma_FD, Sigma_F]])

    E = np.hstack([E_D, E_F])
    return E, Sigma


# 步骤4：贝叶斯方法更新均值和方差-协方差矩阵
def update_mean_covariance(E, Sigma, observed_data):
    return E, Sigma


# 步骤6：更新时变OD需求的均值和方差 √√√
def update_od_demand(E_D, E_D_star, Sigma_D, alpha, rho):
    E_D_new = rho * E_D_star + (1 - rho) * E_D
    Sigma_D = np.diag(alpha * E_D_new)
    return E_D_new, Sigma_D


# 逐步算法主函数
def stepwise_algorithm(I, T, K_list, alpha, rho, n_max, omega):
    # 步骤1：从历史数据中获得时变OD的先验分布
    n = 0

    # 初始化OD需求均值 --> shape[I*T,]
    E_D = np.random.rand(I * T)
    print("初始E(D)",E_D.shape)

    # 初始化方差矩阵 --> shape[I*T,]
    Sigma_D = np.diag(alpha * E_D)
    print("初始σ(D)",Sigma_D.shape)


    while True:
        # 步骤2：求解动态用户均衡
        P = solve_due(E_D, I, T, K_list)
        print("路径选择比例矩阵P",P.shape)

        # 步骤3：计算均值和方差-协方差矩阵 √√√
        E, Sigma = calculate_mean_covariance(E_D, Sigma_D, P)
        print("E(D,F)和σ(D,F)",E.shape,Sigma.shape)

        # 步骤4：用贝叶斯方法更新均值和方差-协方差矩阵
        observed_data = np.random.rand(I * T + sum(K_list) * T)  # 模拟观测数据
        print("观测数据",observed_data.shape)

        E, Sigma = update_mean_covariance(E, Sigma, observed_data)
        print("贝叶斯更新后的E(D,F)和σ(D,F)", E.shape, Sigma.shape)

        E_D_star = E[:I * T]
        print("当前轮预测的E(D)",E_D_star.shape)

        # 步骤5：收敛性检验
        error = np.sum((E_D - E_D_star) ** 2)
        n += 1
        print("当前轮误差",error)
        if n == n_max or error < omega:
            break

        print("进入")
        # 步骤6：更新时变OD需求的均值和方差 √√√
        E_D, Sigma_D = update_od_demand(E_D, E_D_star, Sigma_D, alpha, rho)

    return E_D, Sigma_D


# 参数设置
I = 3  # O - D对数量
T = 4  # 时间间隔数量
K_list = [3, 3, 3]  # 每个O - D对的路径数量列表
alpha = 0.1  # 变异系数
rho = 0.5  # 松弛因子
n_max = 10  # 最大迭代次数
omega = 1e-6  # 收敛阈值

# 运行算法
E_D_final, Sigma_D_final = stepwise_algorithm(I, T, K_list, alpha, rho, n_max, omega)
print("最终OD需求均值:", E_D_final.shape)
# print("最终OD需求均值:", E_D_final)
print("最终OD需求方差矩阵:", Sigma_D_final.shape)
# print("最终OD需求方差矩阵:", Sigma_D_final)
