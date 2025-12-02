import numpy as np
import scipy.stats as stats

# 模拟生成数据
np.random.seed(42)
n = 100  # 样本数量
d = 5    # 输入变量的维度
# 生成输入需求 d_i
D = np.random.randn(n, d)
# 真实的系数向量
true_beta = np.random.randn(d)
# 噪声的标准差
sigma_true = 1.0
# 生成观测到的交通流量计数 v_i
epsilon = np.random.normal(0, sigma_true, n)
V = np.dot(D, true_beta) + epsilon

# 步骤2：确定先验分布的参数
# 先验分布中 beta 的均值
beta_0 = np.linalg.inv(D.T @ D) @ D.T @ V
# 先验分布中 beta 的协方差矩阵
sigma_0_squared = ((V - D @ beta_0).T @ (V - D @ beta_0)) / (n - 1)
Sigma_0 = np.linalg.inv(D.T @ D) * sigma_0_squared
# 先验分布中 1/sigma^2 的形状参数
v_0 = 1
# 先验分布中 1/sigma^2 的尺度参数
v_0_sigma_0_squared = v_0 * sigma_0_squared

# 步骤3和4：使用Gibb's采样器进行参数估计
num_iterations = 1000  # 采样迭代次数
burn_in = 200  # 燃烧期
beta_samples = np.zeros((num_iterations, d))
sigma_squared_samples = np.zeros(num_iterations)

# 初始化参数
beta_current = beta_0
sigma_squared_current = sigma_0_squared

for i in range(num_iterations):
    # 采样 beta
    Sigma_beta = np.linalg.inv(np.linalg.inv(Sigma_0) + D.T @ D / sigma_squared_current)
    E_beta = Sigma_beta @ (np.linalg.inv(Sigma_0) @ beta_0 + D.T @ V / sigma_squared_current)
    beta_current = np.random.multivariate_normal(E_beta, Sigma_beta)

    # 采样 sigma^2
    SSR_beta = (V - D @ beta_current).T @ (V - D @ beta_current)
    shape = (v_0 + n) / 2
    scale = (v_0_sigma_0_squared + SSR_beta) / 2
    sigma_squared_current = 1 / np.random.gamma(shape, scale=1/scale)

    beta_samples[i] = beta_current
    sigma_squared_samples[i] = sigma_squared_current

# 去除燃烧期
beta_estimated = np.mean(beta_samples[burn_in:], axis=0)
sigma_squared_estimated = np.mean(sigma_squared_samples[burn_in:])

print("估计的 beta:", beta_estimated)
print("估计的 sigma^2:", sigma_squared_estimated)