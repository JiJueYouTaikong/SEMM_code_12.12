import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns


# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 定义 Dropout 层，防止过拟合
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置向量，从 0 到 max_len - 1
        position = torch.arange(max_len).unsqueeze(1)
        # 计算用于正弦和余弦函数的分母项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 初始化位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 为偶数索引的列设置正弦值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 为奇数索引的列设置余弦值
        pe[:, 1::2] = torch.cos(position * div_term)
        # 在第 0 维添加一个维度，使形状变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 将位置编码矩阵注册为缓冲区，不参与模型参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码矩阵加到输入上，只取与输入序列长度相同的部分
        x = x + self.pe[:, :x.size(1)]
        # 通过 Dropout 层
        return self.dropout(x)


# 基于 Transformer 的 OD 模型
class ODModel(nn.Module):
    def __init__(self, N, F, d_model=128, nhead=8, num_layers=3, dropout=0):
        # 调用父类的构造函数
        super(ODModel, self).__init__()
        # 位置数量
        self.N = N
        # 每个位置的特征数量
        self.F = F
        # 模型的隐藏维度
        self.d_model = d_model

        # 输入嵌入层，将输入特征 F 映射到 d_model 维空间
        self.input_embedding = nn.Linear(F, d_model)

        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        # 定义 Transformer 编码器，包含 num_layers 层编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层，将 Transformer 编码器的输出投影到 N 维空间
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),  # 线性层
            nn.ReLU(),  # 激活函数
            nn.Linear(d_model, N)  # 输出 N 个值，用于每个位置
        )

        # 初始化模型的权重
        self._reset_parameters()

    def _reset_parameters(self):
        # 遍历模型的所有参数
        for p in self.parameters():
            # 如果参数的维度大于 1
            if p.dim() > 1:
                # 使用 Xavier 均匀分布初始化参数
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x 的形状为 [batch_size, N, F]
        # 获取批量大小
        batch_size = x.size(0)

        # 输入嵌入操作，将输入特征映射到 d_model 维空间
        x = self.input_embedding(x)  # 形状变为 [batch_size, N, d_model]

        # 添加位置编码
        x = self.positional_encoding(x)

        # 通过 Transformer 编码器进行特征提取和转换
        x = self.transformer_encoder(x)  # 形状仍为 [batch_size, N, d_model]

        # 投影到输出空间
        x = self.output_projection(x)  # 形状变为 [batch_size, N, N]

        # 重塑为 OD 矩阵
        od_matrix = x.view(batch_size, self.N, self.N)

        return od_matrix



def load_data():
    set_seed(42)


    speed = np.load('../data/Speed_完整批处理_3.17_Final.npy')
    od = np.load('../data/OD_完整批处理_3.17_Final.npy')

    T, N = speed.shape


    x_random = np.random.normal(loc=0.05, scale=0.01, size=(T, N))
    x_random = np.clip(x_random, 0, 0.1)


    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # Temporal features
    od_train_departures = np.sum(od_train, axis=-1)
    mean_speed = np.mean(speed_train, axis=0)
    mean_departures = np.mean(od_train_departures, axis=0)
    temporal = mean_departures - mean_speed

    temporal_expanded_train = np.tile(temporal, (speed_train.shape[0], 1))
    temporal_expanded_val = np.tile(temporal, (speed_val.shape[0], 1))
    temporal_expanded_test = np.tile(temporal, (speed_test.shape[0], 1))

    # Frequency features
    speed_freq = np.load('../data/速度的周期状态_对应25.1.14的速度数据集.npy')
    od_freq = np.load('../data/OD的周期状态_对应25.1.14的OD数据集.npy')
    freq = od_freq - speed_freq

    freq_expanded_train = np.tile(freq, (speed_train.shape[0], 1))
    freq_expanded_val = np.tile(freq, (speed_val.shape[0], 1))
    freq_expanded_test = np.tile(freq, (speed_test.shape[0], 1))

    # Combine features
    x_train = np.stack([speed_train, temporal_expanded_train, freq_expanded_train, x_random_train], axis=-1)
    x_val = np.stack([speed_val, temporal_expanded_val, freq_expanded_val, x_random_val], axis=-1)
    x_test = np.stack([speed_test, temporal_expanded_test, freq_expanded_test, x_random_test], axis=-1)

    # Normalize
    scaler = MinMaxScaler()
    train_3 = x_train[..., :3]
    val_3 = x_val[..., :3]
    test_3 = x_test[..., :3]

    x_train[..., :3] = scaler.fit_transform(train_3.reshape(-1, 3)).reshape(train_3.shape)
    x_val[..., :3] = scaler.transform(val_3.reshape(-1, 3)).reshape(val_3.shape)
    x_test[..., :3] = scaler.transform(test_3.reshape(-1, 3)).reshape(test_3.shape)

    # Use only speed feature for transformer input
    train_data = x_train[..., 0:1]  # Using only speed feature (N, 1)
    val_data = x_val[..., 0:1]
    test_data = x_test[..., 0:1]

    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


# Training function
def train_model(model, train_loader, val_loader, epochs=100, patience=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ckpt/best_transformer_model.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


# Evaluation metrics
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


# Testing function
def test_model(model, test_loader,lr=0,log_filename=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_transformer_model.pth"))

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
            f"Lr = {lr},Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f} CPC:{cpc_total:.4f} JSD:{jsd_total:.4f}\n")



# Main function
def main():
    # 保存日志文件
    log_filename = f"log/Transformer_完整批处理调参.log"

    # 定义学习率列表
    lr_list = [0.01, 0.005, 0.004, 0.003, 0.002, 0.001,
               0.0005, 0.0004 , 0.0003 ,0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    lr_list = [0.001]  # 是否当前最佳 YES 6.20
    # 遍历学习率列表
    for lr in lr_list:
        print(f"当前学习率: {lr}")

        set_seed(42)

        # Load data
        train_loader, val_loader, test_loader = load_data()

        # Initialize model
        model = ODModel(N=110, F=1, d_model=128, nhead=8, num_layers=3)

        # Train model
        train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=lr)

        # 测试模型
        test_model(model, test_loader, lr=lr, log_filename=log_filename)


if __name__ == "__main__":
    main()