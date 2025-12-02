#
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import gpytorch
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import os
# from typing import Tuple, Dict
#
# # 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
#
#
# class ODGPModel(gpytorch.models.ExactGP):
#     """高斯过程模型用于OD矩阵估计"""
#
#     def __init__(self, train_x, train_y, likelihood, input_dim):
#         super(ODGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# def load_data(file_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     加载或生成模拟数据
#
#     参数:
#         file_path: 数据文件路径，如果为None则生成模拟数据
#
#     返回:
#         speed_data: 速度数据 [T, N]
#         departure_data: 出发总量数据 [T, N]
#         od_data: OD矩阵数据 [T, N, N]
#     """
#     if file_path and os.path.exists(file_path):
#         # 从文件加载数据
#         data = np.load(file_path)
#         speed_data = data['speed']
#         departure_data = data['departure']
#         od_data = data['od']
#     else:
#         # 生成模拟数据
#         print("数据文件不存在，生成模拟数据...")
#         np.random.seed(42)
#         T = 100  # 时间步数
#         N = 5  # 区域数
#
#         # 生成OD矩阵 [T, N, N]
#         od_data = np.random.rand(T, N, N) * 100
#         # 确保对角元素为0（区域内流量）
#         for t in range(T):
#             np.fill_diagonal(od_data[t], 0)
#
#         # 计算出发总量 [T, N]
#         departure_data = np.sum(od_data, axis=2)
#
#         # 生成速度数据 [T, N]，与OD矩阵有一定关系
#         speed_data = 60 - departure_data / 10 + np.random.normal(0, 5, size=(T, N))
#         speed_data = np.maximum(5, speed_data)  # 确保速度不小于5
#
#     return speed_data, departure_data, od_data
#
#
# def preprocess_data(
#         speed_data: np.ndarray,
#         departure_data: np.ndarray,
#         od_data: np.ndarray,
#         train_ratio: float = 0.6,
#         val_ratio: float = 0.2,
#         test_ratio: float = 0.2
# ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
#     """
#     数据预处理和划分
#
#     参数:
#         speed_data: 速度数据 [T, N]
#         departure_data: 出发总量数据 [T, N]
#         od_data: OD矩阵数据 [T, N, N]
#         train_ratio: 训练集比例
#         val_ratio: 验证集比例
#         test_ratio: 测试集比例
#
#     返回:
#         包含训练集、验证集和测试集的字典
#     """
#     T, N = speed_data.shape
#
#     # 确保比例之和为1
#     assert train_ratio + val_ratio + test_ratio == 1.0
#
#     # 数据标准化 - 只使用速度数据
#     speed_scaler = StandardScaler()
#
#     speed_data_flat = speed_data.reshape(-1, N)
#     speed_data_normalized = speed_scaler.fit_transform(speed_data_flat)
#
#     # 重塑回原始形状
#     speed_data_normalized = speed_data_normalized.reshape(T, N)
#
#     # 准备输入特征：只使用速度数据
#     features = speed_data_normalized  # [T, N]
#
#     # 准备目标：将OD矩阵展平
#     targets = od_data.reshape(T, N * N)  # [T, N*N]
#
#     # 划分数据集
#     indices = np.arange(T)
#     train_idx, test_idx = train_test_split(
#         indices, test_size=test_ratio + val_ratio, random_state=42
#     )
#     val_idx, test_idx = train_test_split(
#         test_idx, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
#     )
#
#     # 创建数据集字典
#     datasets = {}
#     for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
#         X = features[idx]
#         y = targets[idx]
#
#         # 转换为PyTorch张量
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32)
#
#         # 为每个OD对创建单独的模型
#         datasets[split] = (X_tensor, y_tensor)
#
#     return datasets, speed_scaler
#
#
# def train_gp_models(
#         datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#         input_dim: int,
#         n_pairs: int,
#         device: torch.device = torch.device('cpu')
# ) -> Tuple[Dict[int, ODGPModel], Dict[int, gpytorch.likelihoods.GaussianLikelihood]]:
#     """
#     训练高斯过程模型
#
#     参数:
#         datasets: 数据集字典
#         input_dim: 输入特征维度
#         n_pairs: OD对数量 (N*N)
#         device: 计算设备
#
#     返回:
#         models: 训练好的模型字典
#         likelihoods: 似然函数字典
#     """
#     X_train, y_train = datasets['train']
#     X_train, y_train = X_train.to(device), y_train.to(device)
#
#     models = {}
#     likelihoods = {}
#
#     # 为每个OD对训练一个GP模型
#     for pair_idx in range(n_pairs):
#         print(f"训练OD对 {pair_idx + 1}/{n_pairs}...")
#
#         # 提取当前OD对的目标值
#         y_pair = y_train[:, pair_idx]
#
#         # 初始化 likelihood 和 model
#         likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
#         model = ODGPModel(X_train, y_pair, likelihood, input_dim).to(device)
#
#         # 训练模型
#         model.train()
#         likelihood.train()
#
#         # 使用 Adam 优化器
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#
#         # "Loss" for GPs - the marginal log likelihood
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#
#         # 训练迭代
#         n_iter = 50
#         for i in range(n_iter):
#             optimizer.zero_grad()
#             output = model(X_train)
#             loss = -mll(output, y_pair)
#             loss.backward()
#             optimizer.step()
#
#             if (i + 1) % 10 == 0:
#                 print(f"OD对 {pair_idx + 1}, 迭代 {i + 1}/{n_iter}, 损失: {loss.item():.4f}")
#
#         models[pair_idx] = model
#         likelihoods[pair_idx] = likelihood
#
#     return models, likelihoods
#
#
# def evaluate_models(
#         models: Dict[int, ODGPModel],
#         likelihoods: Dict[int, gpytorch.likelihoods.GaussianLikelihood],
#         datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#         n_pairs: int,
#         device: torch.device = torch.device('cpu')
# ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     评估模型性能
#
#     参数:
#         models: 模型字典
#         likelihoods: 似然函数字典
#         datasets: 数据集字典
#         n_pairs: OD对数量
#         device: 计算设备
#
#     返回:
#         包含各数据集预测结果、标准差和真实值的字典
#     """
#     results = {}
#
#     for split, (X, y_true) in datasets.items():
#         X, y_true = X.to(device), y_true.to(device)
#         n_samples = X.shape[0]
#
#         # 初始化预测结果和标准差数组
#         y_pred = torch.zeros(n_samples, n_pairs, device=device)
#         y_std = torch.zeros(n_samples, n_pairs, device=device)
#
#         # 对每个OD对进行预测
#         for pair_idx in range(n_pairs):
#             model = models[pair_idx]
#             likelihood = likelihoods[pair_idx]
#
#             # 设置为评估模式
#             model.eval()
#             likelihood.eval()
#
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 observed_pred = likelihood(model(X))
#                 y_pred[:, pair_idx] = observed_pred.mean
#                 y_std[:, pair_idx] = observed_pred.stddev
#
#         # 转换为numpy数组
#         results[split] = (
#             y_pred.cpu().numpy(),
#             y_std.cpu().numpy(),
#             y_true.cpu().numpy()
#         )
#
#     return results
#
#
# def visualize_results(
#         results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
#         n_pairs: int,
#         N: int,
#         split: str = 'test',
#         sample_idx: int = 0
# ) -> None:
#     """
#     可视化预测结果和置信区间
#
#     参数:
#         results: 评估结果字典
#         n_pairs: OD对数量
#         N: 区域数
#         split: 要可视化的数据集
#         sample_idx: 要可视化的样本索引
#     """
#     y_pred, y_std, y_true = results[split]
#
#     # 随机选择几个OD对进行可视化
#     np.random.seed(42)
#     selected_pairs = np.random.choice(n_pairs, min(9, n_pairs), replace=False)
#
#     # 创建子图
#     fig, axes = plt.subplots(3, 3, figsize=(15, 12))
#     axes = axes.flatten()
#
#     for i, pair_idx in enumerate(selected_pairs):
#         # 计算OD对的起点和终点
#         origin = pair_idx // N
#         destination = pair_idx % N
#
#         # 获取预测值、标准差和真实值
#         pred = y_pred[:, pair_idx]
#         std = y_std[:, pair_idx]
#         true = y_true[:, pair_idx]
#
#         # 计算95%置信区间
#         lower = pred - 1.96 * std
#         upper = pred + 1.96 * std
#
#         # 绘制预测结果和置信区间
#         axes[i].plot(true, label='真实值', color='blue')
#         axes[i].plot(pred, label='预测值', color='red')
#         axes[i].fill_between(range(len(pred)), lower, upper, color='gray', alpha=0.2, label='95%置信区间')
#
#         axes[i].set_title(f'OD对 ({origin}→{destination})')
#         axes[i].set_xlabel('时间步')
#         axes[i].set_ylabel('流量')
#         axes[i].legend()
#         axes[i].grid(True)
#
#     plt.tight_layout()
#     plt.savefig(f'{split}_prediction_intervals.png')
#     plt.show()
#
#     # 可视化每个时间步的平均标准差
#     plt.figure(figsize=(10, 6))
#     mean_std = np.mean(y_std, axis=1)
#     plt.plot(mean_std)
#     plt.title('每个时间步的平均标准差')
#     plt.xlabel('时间步')
#     plt.ylabel('平均标准差')
#     plt.grid(True)
#     plt.savefig('time_step_std.png')
#     plt.show()
#
#
# def main():
#     """主函数"""
#     # 加载数据
#     speed_data, departure_data, od_data = load_data()
#     T, N = speed_data.shape
#
#     # 数据预处理和划分 - 只使用速度数据
#     datasets, speed_scaler = preprocess_data(speed_data, departure_data, od_data)
#
#     # 设置计算设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 获取输入维度和OD对数量
#     X_train, _ = datasets['train']
#     input_dim = X_train.shape[1]
#     n_pairs = N * N
#
#     # 训练模型
#     models, likelihoods = train_gp_models(datasets, input_dim, n_pairs, device)
#
#     # 评估模型
#     results = evaluate_models(models, likelihoods, datasets, n_pairs, device)
#
#     # 可视化结果
#     visualize_results(results, n_pairs, N, split='train', sample_idx=10)
#     visualize_results(results, n_pairs, N, split='val', sample_idx=5)
#     visualize_results(results, n_pairs, N, split='test', sample_idx=3)
#
#     # 保存每个时间步的标准差
#     for split in ['train', 'val', 'test']:
#         _, y_std, _ = results[split]
#         time_step_std = np.mean(y_std, axis=1)
#         np.savetxt(f'{split}_time_step_std.txt', time_step_std)
#         print(f"{split}集的时间步标准差已保存到 {split}_time_step_std.txt")
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, Dict

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class ODGPModel(gpytorch.models.ExactGP):
    """高斯过程模型用于OD矩阵估计"""

    def __init__(self, train_x, train_y, likelihood, input_dim):
        super(ODGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_data(file_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载或生成模拟数据

    参数:
        file_path: 数据文件路径，如果为None则生成模拟数据

    返回:
        speed_data: 速度数据 [T, N]
        departure_data: 出发总量数据 [T, N]
        od_data: OD矩阵数据 [T, N, N]
    """
    if file_path and os.path.exists(file_path):
        # 从文件加载数据
        data = np.load(file_path)
        speed_data = data['speed']
        departure_data = data['departure']
        od_data = data['od']
    else:
        # 生成模拟数据
        print("数据文件不存在，生成模拟数据...")
        np.random.seed(42)
        T = 100  # 时间步数
        N = 5  # 区域数

        # 生成OD矩阵 [T, N, N]
        od_data = np.random.rand(T, N, N) * 100
        # 确保对角元素为0（区域内流量）
        for t in range(T):
            np.fill_diagonal(od_data[t], 0)

        # 计算出发总量 [T, N]
        departure_data = np.sum(od_data, axis=2)

        # 生成速度数据 [T, N]，与OD矩阵有一定关系
        # speed_data = 60 - departure_data / 10 + np.random.normal(0, 5, size=(T, N))
        # speed_data = np.maximum(5, speed_data)  # 确保速度不小于5
        speed_data = np.random.randn(T, N) * 70

    return speed_data, departure_data, od_data


def preprocess_data(
        speed_data: np.ndarray,
        departure_data: np.ndarray,
        od_data: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    数据预处理和划分

    参数:
        speed_data: 速度数据 [T, N]
        departure_data: 出发总量数据 [T, N]
        od_data: OD矩阵数据 [T, N, N]
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    返回:
        包含训练集、验证集和测试集的字典
    """
    T, N = speed_data.shape

    # 确保比例之和为1
    assert train_ratio + val_ratio + test_ratio == 1.0

    # 数据标准化 - 只使用速度数据
    speed_scaler = StandardScaler()

    speed_data_flat = speed_data.reshape(-1, N)
    speed_data_normalized = speed_scaler.fit_transform(speed_data_flat)

    # 重塑回原始形状
    speed_data_normalized = speed_data_normalized.reshape(T, N)

    # 准备输入特征：只使用速度数据
    features = speed_data_normalized  # [T, N]

    # 准备目标：将OD矩阵展平
    targets = od_data.reshape(T, N * N)  # [T, N*N]

    # 划分数据集
    indices = np.arange(T)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio + val_ratio, random_state=42
    )
    val_idx, test_idx = train_test_split(
        test_idx, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
    )

    # 创建数据集字典
    datasets = {}
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        X = features[idx]
        y = targets[idx]

        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 为每个OD对创建单独的模型
        datasets[split] = (X_tensor, y_tensor)

    return datasets, speed_scaler


def train_gp_models(
        datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        input_dim: int,
        n_pairs: int,
        device: torch.device = torch.device('cpu')
) -> Tuple[Dict[int, ODGPModel], Dict[int, gpytorch.likelihoods.GaussianLikelihood]]:
    """
    训练高斯过程模型

    参数:
        datasets: 数据集字典
        input_dim: 输入特征维度
        n_pairs: OD对数量 (N*N)
        device: 计算设备

    返回:
        models: 训练好的模型字典
        likelihoods: 似然函数字典
    """
    X_train, y_train = datasets['train']
    X_train, y_train = X_train.to(device), y_train.to(device)

    models = {}
    likelihoods = {}

    # 为每个OD对训练一个GP模型
    for pair_idx in range(n_pairs):
        print(f"训练OD对 {pair_idx + 1}/{n_pairs}...")

        # 提取当前OD对的目标值
        y_pair = y_train[:, pair_idx]

        # 初始化 likelihood 和 model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ODGPModel(X_train, y_pair, likelihood, input_dim).to(device)

        # 训练模型
        model.train()
        likelihood.train()

        # 使用 Adam 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # 训练迭代
        n_iter = 50
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_pair)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"OD对 {pair_idx + 1}, 迭代 {i + 1}/{n_iter}, 损失: {loss.item():.4f}")

        models[pair_idx] = model
        likelihoods[pair_idx] = likelihood

    return models, likelihoods


def evaluate_models(
        models: Dict[int, ODGPModel],
        likelihoods: Dict[int, gpytorch.likelihoods.GaussianLikelihood],
        datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        n_pairs: int,
        device: torch.device = torch.device('cpu')
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    评估模型性能

    参数:
        models: 模型字典
        likelihoods: 似然函数字典
        datasets: 数据集字典
        n_pairs: OD对数量
        device: 计算设备

    返回:
        包含各数据集预测结果、标准差和真实值的字典
    """
    results = {}

    for split, (X, y_true) in datasets.items():
        X, y_true = X.to(device), y_true.to(device)
        n_samples = X.shape[0]

        # 初始化预测结果和标准差数组
        y_pred = torch.zeros(n_samples, n_pairs, device=device)
        y_std = torch.zeros(n_samples, n_pairs, device=device)

        # 对每个OD对进行预测
        for pair_idx in range(n_pairs):
            model = models[pair_idx]
            likelihood = likelihoods[pair_idx]

            # 设置为评估模式
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(X))
                y_pred[:, pair_idx] = observed_pred.mean
                y_std[:, pair_idx] = observed_pred.stddev

        # 转换为numpy数组
        results[split] = (
            y_pred.cpu().numpy(),
            y_std.cpu().numpy(),
            y_true.cpu().numpy()
        )

    return results


def visualize_results(
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_pairs: int,
        N: int,
        split: str = 'test',
        sample_idx: int = 0
) -> None:
    """
    可视化预测结果和置信区间

    参数:
        results: 评估结果字典
        n_pairs: OD对数量
        N: 区域数
        split: 要可视化的数据集
        sample_idx: 要可视化的样本索引
    """
    y_pred, y_std, y_true = results[split]

    # 随机选择几个OD对进行可视化
    np.random.seed(42)
    selected_pairs = np.random.choice(n_pairs, min(9, n_pairs), replace=False)

    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, pair_idx in enumerate(selected_pairs):
        # 计算OD对的起点和终点
        origin = pair_idx // N
        destination = pair_idx % N

        # 获取预测值、标准差和真实值
        pred = y_pred[:, pair_idx]
        std = y_std[:, pair_idx]
        true = y_true[:, pair_idx]

        # 计算95%置信区间
        lower = pred - 1.96 * std
        upper = pred + 1.96 * std

        # 绘制预测结果和置信区间
        axes[i].plot(true, label='真实值', color='blue')
        axes[i].plot(pred, label='预测值', color='red')
        axes[i].fill_between(range(len(pred)), lower, upper, color='gray', alpha=0.2, label='95%置信区间')

        axes[i].set_title(f'OD对 ({origin}→{destination})')
        axes[i].set_xlabel('时间步')
        axes[i].set_ylabel('流量')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(f'{split}_prediction_intervals.png')
    plt.show()

    # 随机选择4个OD对来展示标准差折线图
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    selected_std_pairs = np.random.choice(n_pairs, 4, replace=False)

    plt.figure(figsize=(12, 8))
    for i, pair_idx in enumerate(selected_std_pairs):
        # 计算OD对的起点和终点
        origin = pair_idx // N
        destination = pair_idx % N

        # 获取该OD对的标准差
        std = y_std[:, pair_idx]

        # 绘制标准差折线图
        plt.plot(std, label=f'OD对 ({origin}→{destination})')

    plt.title('随机选择的4个OD对的标准差随时间变化')
    plt.xlabel('时间步')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{split}_selected_od_std.png')
    plt.show()


def main():
    """主函数"""
    # 加载数据
    speed_data, departure_data, od_data = load_data()
    T, N = speed_data.shape

    # 数据预处理和划分 - 只使用速度数据
    datasets, speed_scaler = preprocess_data(speed_data, departure_data, od_data)

    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取输入维度和OD对数量
    X_train, _ = datasets['train']
    input_dim = X_train.shape[1]
    n_pairs = N * N

    # 训练模型
    models, likelihoods = train_gp_models(datasets, input_dim, n_pairs, device)

    # 评估模型
    results = evaluate_models(models, likelihoods, datasets, n_pairs, device)

    # 可视化结果
    visualize_results(results, n_pairs, N, split='train', sample_idx=10)
    visualize_results(results, n_pairs, N, split='val', sample_idx=5)
    visualize_results(results, n_pairs, N, split='test', sample_idx=3)

    # 保存每个时间步的标准差
    for split in ['train', 'val', 'test']:
        _, y_std, _ = results[split]
        time_step_std = np.mean(y_std, axis=1)
        # np.savetxt(f'{split}_time_step_std.txt', time_step_std)
        # print(f"{split}集的时间步标准差已保存到 {split}_time_step_std.txt")


if __name__ == "__main__":
    main()