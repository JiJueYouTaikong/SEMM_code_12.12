import torch
import gpytorch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 假设 T=100, N=5，速度 X 为 [T, N]，OD Y 为 [T, N, N]
T, N = 100, 5
X = torch.randn(T, N)                     # 输入速度数据
Y = torch.randn(T, N * N)                # 输出 OD，展开成 [T, N*N]

# 拆分训练 / 测试
train_X = X[:80]
train_Y = Y[:80]
test_X = X[80:]
test_Y = Y[80:]

# 数据维度
input_dim = N
output_dim = N * N

# 使用 GPyTorch 的多任务 GP 实现
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, input_dim):
        # 使用独立任务的变分分布和策略
        inducing_points = torch.randn(num_tasks, 64, input_dim)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            num_inducing_points=64, batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # 每个输出任务一个核
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # x: [batch, input_dim]
        # 输出 shape: [num_tasks, batch]
        mean_x = self.mean_module(x).transpose(0, 1)  # [num_tasks, batch]
        covar_x = self.covar_module(x).transpose(0, 1)  # [num_tasks, batch, batch]
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, num_tasks, input_dim):
        super().__init__(None)
        self.model = MultitaskGPModel(num_tasks, input_dim)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, x):
        return self.model(x)


# 初始化模型与优化器
model = MultitaskGPModel(output_dim, input_dim)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_Y.size(0))

# 训练
num_epochs = 200
for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, train_Y.T)  # 注意维度：Y 需要 [output_dim, T]
    loss.backward()
    print(f'Iter {i + 1}/{num_epochs} - Loss: {loss.item():.3f}')
    optimizer.step()

# 预测
model.eval()
likelihood.eval()
with torch.no_grad():
    preds = likelihood(model(test_X))
    mean = preds.mean.T  # [T_test, output_dim]
    std = preds.stddev.T

    # 选择第一个 OD 对可视化
    od_idx = 0
    plt.figure(figsize=(10, 5))
    plt.plot(test_Y[:, od_idx].numpy(), 'k*', label='True')
    plt.plot(mean[:, od_idx].numpy(), 'b', label='Predicted')
    plt.fill_between(
        torch.arange(mean.size(0)).numpy(),
        (mean[:, od_idx] - 2 * std[:, od_idx]).numpy(),
        (mean[:, od_idx] + 2 * std[:, od_idx]).numpy(),
        alpha=0.3,
        label='Confidence'
    )
    plt.legend()
    plt.title(f'OD Pair {od_idx} Prediction')
    plt.show()
