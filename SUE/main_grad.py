import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.autograd import Function

from SUE.utils.DTA import run_dta_logit

def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    # 计算 MAPE，避免除以零
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(
            torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()



# 参数设置
capacity = 500
alpha = 0.15
beta = 4
lambda_param = 1
max_iter = 1
tol = 1e-3

# 加载并转换为PyTorch张量（需确保数据在CPU上）
adj_matrix = np.load('../data/adj110_3.17.npy')
dist_matrix = np.load('../data/dist110_3.17.npy')
adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32, requires_grad=False)
dist_matrix_tensor = torch.tensor(dist_matrix, dtype=torch.float32, requires_grad=False)


# 自定义autograd Function
class DTALogitFunction(Function):
    @staticmethod
    def forward(ctx, od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol):
        # 保存参数供反向传播使用
        ctx.adj_matrix = adj_matrix.numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
        ctx.dist_matrix = dist_matrix.numpy() if isinstance(dist_matrix, torch.Tensor) else dist_matrix
        ctx.lambda_param = lambda_param
        ctx.max_iter = max_iter
        ctx.tol = tol

        # 转换OD矩阵为numpy并运行DTA
        od_matrix_np = od_matrix.detach().cpu().numpy()
        flow_pred_np = run_dta_logit(od_matrix_np, ctx.adj_matrix, ctx.dist_matrix, lambda_param, max_iter, tol)
        flow_pred = torch.tensor(flow_pred_np, dtype=od_matrix.dtype, device=od_matrix.device)

        # 保存输入输出供反向传播
        ctx.save_for_backward(od_matrix, flow_pred)
        return flow_pred

    @staticmethod
    def backward(ctx, grad_output):
        # 简化梯度计算：假设雅可比矩阵为单位阵（实际需根据模型调整）
        od_matrix, flow_pred = ctx.saved_tensors
        grad_input = grad_output.clone()  # 此处应替换为实际梯度计算
        return grad_input, None, None, None, None, None


# 修改后的交通分配函数
def dta_model(od_matrix):
    return DTALogitFunction.apply(od_matrix, adj_matrix_tensor, dist_matrix_tensor, lambda_param, max_iter, tol)


# 双层模型（保持不变）
class BilevelFramework(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.ode = nn.Parameter(torch.randn(N, N))

    def forward(self):
        return dta_model(self.ode)


# 数据集
N = 110
num_samples = 3
# 观测的交通流量
flow = np.load('data/Link_flow_TNN_3.17.npy')
od = np.load('data/OD_完整批处理_3.17_Final.npy')
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)

print(flow.shape)
print(od.shape)

# 增量梯度下降优化
num_epochs = 2

rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0

for i in range(num_samples):
    # print(f"样本{i}:")
    # 对每一组样本随机初始化估计的OD矩阵
    model = BilevelFramework(N)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):

        print(f"model.ode:{model.ode}")

        optimizer.zero_grad()

        predicted_flow = model()

        flow_tensor = torch.tensor(flow[i], dtype=torch.float32)  # 将 numpy 数组转换为 torch.Tensor

        loss = criterion(predicted_flow, flow_tensor)

        loss.backward()

        optimizer.step()

        print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


    loss_total += loss.item()
    pred_od = model.ode
    print("------------------------")
    print(f"Final pred_od:{pred_od}")
    print(f"Final true od:{od[i]}")
    #
    # print("------------------------")
    # print(f"Final pred_od:{pred_od[60:68,60:68]}")
    # print(f"Final true od:{od[i][60:68,60:68]}")

    od_tensor = torch.tensor(od[i], dtype=torch.float32)  # 将 numpy 数组转换为 torch.Tensor
    rmse, mae, mape = calculate_rmse_mae(pred_od, od_tensor)

    rmse_total += rmse
    mae_total += mae
    mape_total += mape

rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"Test loss:{loss_test} RMSE:{rmse_test} MAE:{mae_test} MAPE:{mape_test}")