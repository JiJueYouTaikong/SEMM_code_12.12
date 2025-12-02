import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
# from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据准备
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val




# 定义模型
class ODModel(nn.Module):
    def __init__(self, N):
        super(ODModel, self).__init__()
        self.N = N

        layers = []
        input_dim = N

        # 前 6 层：256 维
        for i in range(6):
            layers.append(nn.Linear(input_dim, 256))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            input_dim = 256

        # 后 9 层：128 维
        for i in range(8):  # 前 8 层是 128 -> 128
            layers.append(nn.Linear(input_dim, 128))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            input_dim = 128

        # 第 15 层（最后一层隐藏层）：128 -> 输出 N*N
        layers.append(nn.Linear(input_dim, N * N))

        # 封装为 Sequential
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: 输入速度 [B, N]
        :return: od矩阵 [B, N, N]
        '''
        batch_size, N = x.shape
        od_matrix_flat = self.mlp(x)  # [B, N*N]
        od_matrix = od_matrix_flat.view(batch_size, N, N)
        return od_matrix


# 训练过程
def train_model(model, train_loader, val_loader, epochs=100, patience=10,learning_rate=0.001):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    best_val_loss = float('inf')
    patience_counter = 0
    N = 110


    # 训练过程
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算训练集的平均损失
        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # 计算验证集的平均损失
        val_loss /= len(val_loader)


        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # ({optimizer.param_groups[0]['lr']:.6f})

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/best_model_DeepGravity.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def calculate_rmse_mae(predictions, targets):
    '''

    :param predictions: [T,N,N] T个时间步上的OD矩阵预测值
    :param targets:  [T,N,N] T个时间步上的OD矩阵真值
    :return:
    '''
    T, N, _ = predictions.shape
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

    # Flatten
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    targ_flat = targets.reshape(targets.shape[0], -1)

    # CPC
    cpc_list = []
    for t in range(pred_flat.shape[0]):
        pred_t = pred_flat[t]
        targ_t = targ_flat[t]
        numerator = 2 * torch.sum(torch.minimum(pred_t, targ_t))
        denominator = torch.sum(pred_t) + torch.sum(targ_t)
        if denominator > 0:
            cpc_list.append((numerator / denominator).item())
        else:
            cpc_list.append(1.0)
    cpc = sum(cpc_list) / len(cpc_list)

    # JSD
    jsd_list = []
    min_val = 1e-8  # 安全裁剪阈值

    for t in range(pred_flat.shape[0]):
        pred_t = pred_flat[t]
        targ_t = targ_flat[t]

        # 构造分布
        pred_dist = (pred_t + min_val) / (torch.sum(pred_t) + min_val * pred_t.numel())
        targ_dist = (targ_t + min_val) / (torch.sum(targ_t) + min_val * targ_t.numel())

        # 强制裁剪，防止log(0)
        pred_dist = torch.clamp(pred_dist, min=min_val)
        targ_dist = torch.clamp(targ_dist, min=min_val)
        m = 0.5 * (pred_dist + targ_dist)
        m = torch.clamp(m, min=min_val)

        kl1 = torch.sum(pred_dist * torch.log(pred_dist / m))
        kl2 = torch.sum(targ_dist * torch.log(targ_dist / m))
        jsd_t = 0.5 * (kl1 + kl2)

        if not torch.isnan(jsd_t):
            jsd_list.append(jsd_t.item())
    jsd = sum(jsd_list) / len(jsd_list) if jsd_list else 0.0


    return rmse.item(), mae.item(), mape.item(),cpc,jsd

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子




# 加载数据
def load_data(is_mcm=False):

    log_filename = f"log/DeepGravity_完整批处理_MCM_{is_mcm}.log"

    set_seed(42)

    if not is_mcm:
        speed = np.load('../data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final.npy')
        # 获取数据长度 T
        T, N = speed.shape
        # 按顺序划分数据
        train_size = int(T * 0.6)
        val_size = int(T * 0.2)
    else:
        speed = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final_MCM_60.npy')
        # 获取数据长度 T
        T, N = speed.shape
        # 根据指定的时间步数划分
        test_size = 35
        val_size = 33
        train_size = T - test_size - val_size

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]

    # 输出划分后的数据形状
    print("6:2:2顺序划分的训练集Speed", speed_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集Speed", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集Speed", speed_test.shape, "OD", od_test.shape)

    # 归一化
    scaler = MinMaxScaler()

    train_x_scaler = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    val_x_scaler = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    tes_x_scaler = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)


    train_data = train_x_scaler
    val_data = val_x_scaler
    test_data = tes_x_scaler

    print(train_data[0, 66:82])


    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("归一化后的训练集 shape:", train_data.shape, "OD形状", train_target.shape)
    print("归一化后的验证集 shape:", val_data.shape, "OD形状",val_target.shape)
    print("归一化后的测试集 shape:", test_data.shape, "OD形状", test_target.shape)


    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    batch = 32
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    return train_loader, val_loader, test_loader,log_filename


import time

# 测试
def test_model(model, test_loader,lr=0,log_filename=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_model_DeepGravity.pth"))

    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0
    cpc_total = 0
    jsd_total = 0

    criterion = nn.MSELoss()
    N= 110

    all_real_od = []
    all_pred_od = []

    start_time = time.time()

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            # 设置对角线掩码
            mask = torch.ones_like(targets)
            for i in range(N):
                mask[:, i, i] = 0  # 对角线上的元素设为 0

            print(f"输入:{inputs.shape},标签:{targets.shape}")

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 计算 RMSE 和 MAE
            rmse, mae, mape,cpc,jsd= calculate_rmse_mae(outputs * mask, targets)
            rmse_total += rmse
            mae_total += mae
            mape_total += mape
            cpc_total += cpc
            jsd_total += jsd

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    end_time = time.time()
    infer_time = start_time - end_time


    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)
    cpc_total /= len(test_loader)
    jsd_total /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f} CPC:{cpc_total:.4f} JSD:{jsd_total:.4f}")

    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)

    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Lr = {lr},Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f} CPC:{cpc_total:.4f} JSD:{jsd_total:.4f} Infer time:{infer_time}s\n")


# 主程序
def main():



    # 定义学习率列表
    # lr_list = [ 0.01, 0.005,
    #            0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0002, 0.0001]
    lr_list = [0.0035] #  是否当前最佳 YES 6.20
    lr_list = [0.003]   # MCM 是否当前最佳 YES 6.20
    # 遍历学习率列表
    for lr in lr_list:
        print(f"当前学习率: {lr}")

        train_loader, val_loader, test_loader, log_filename = load_data(is_mcm=True)

        model = ODModel(N=110)

        # 训练模型
        train_model(model, train_loader, val_loader, epochs=2000, patience=40, learning_rate=lr)

        # 测试模型
        test_model(model, test_loader, lr=lr,log_filename=log_filename)


if __name__ == "__main__":
    main()
