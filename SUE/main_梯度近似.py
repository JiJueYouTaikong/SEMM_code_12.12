import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx

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
capacity = 500  # 路段容量
alpha = 0.15  # BPR函数的alpha参数
beta = 4  # BPR函数的beta参数
lambda_param = 1  # Logit模型灵敏度参数
max_iter = 1  # 最大迭代次数
tol = 1e-3  # 收敛阈值

adj_matrix = np.load('../data/adj110_3.17.npy')  # [N, N]
dist_matrix = np.load('../data/dist110_3.17.npy')  # [N, N]


# 损失函数（均方误差）
def loss_function(pred_Flow, true_Flow):
    return np.mean((pred_Flow - true_Flow) ** 2)

# 梯度计算（数值近似）
def compute_gradient(OD, adj, distance, true_Flow, lambda_param, max_iter, tol):
    gradient = np.zeros_like(OD)
    h = 1e-6  # 微小变化量
    for i in range(N):
        print(f"梯度近似：{i}")
        for j in range(N):
            OD_plus_h = OD.copy()
            OD_plus_h[i, j] += h
            Flow_plus_h = dta_model(OD_plus_h)
            loss_plus_h = loss_function(Flow_plus_h, true_Flow)

            OD_minus_h = OD.copy()
            OD_minus_h[i, j] -= h
            Flow_minus_h = dta_model(OD_minus_h)
            loss_minus_h = loss_function(Flow_minus_h, true_Flow)

            gradient[i, j] = (loss_plus_h - loss_minus_h) / (2 * h)
    return gradient

# 交通分配
def dta_model(od_matrix):
    '''
    :param od_matrix:      OD      N,N
    :return:              分配流量  N,N
    '''

    pred_od = run_dta_logit(od_matrix, adj_matrix, dist_matrix, lambda_param, max_iter, tol)


    return pred_od

N = 110
ode = np.random.rand(N, N)



# 数据集
N = 110
num_samples = 1
# 观测的交通流量
flow = np.load('data/Link_flow_TNN_3.17.npy')
od = np.load('data/OD_完整批处理_3.17_Final.npy')
flow = flow[-num_samples:].astype(np.float32)
od = od[-num_samples:].astype(np.float32)

print(flow.shape)
print(od.shape)

# 增量梯度下降优化
num_epochs = 3

rmse_total = 0
mae_total = 0
mape_total = 0
loss_total = 0

for i in range(num_samples):
    # print(f"样本{i}:")
    # 对每一组样本随机初始化估计的OD矩阵

    true_Flow = flow[i]


    for epoch in range(num_epochs):



        # print(f"ode:{ode}")

        pred_flow = dta_model(ode)

        # 计算损失
        loss = loss_function(pred_flow, true_Flow)


        # 计算梯度
        gradient = compute_gradient(ode, adj_matrix, dist_matrix, true_Flow, lambda_param, max_iter, tol)

        # 更新OD矩阵
        ode -= 0.1 * gradient

        print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


    loss_total += loss.item()
    pred_od = ode
    print("------------------------")
    print(f"Final pred_od:{pred_od}")
    print(f"Final true od:{od[i]}")
    #
    # print("------------------------")
    # print(f"Final pred_od:{pred_od[60:68,60:68]}")
    # print(f"Final true od:{od[i][60:68,60:68]}")

    # od_tensor = torch.tensor(od[i], dtype=torch.float32)  # 将 numpy 数组转换为 torch.Tensor
    rmse, mae, mape = calculate_rmse_mae(pred_od, od[i])

    rmse_total += rmse
    mae_total += mae
    mape_total += mape

rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
loss_test = loss_total / num_samples

print(f"Test loss:{loss_test} RMSE:{rmse_test} MAE:{mae_test} MAPE:{mape_test}")