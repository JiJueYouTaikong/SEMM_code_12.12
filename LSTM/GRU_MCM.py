import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
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
from scipy import stats


class GRUModel(nn.Module):  # Changed from LSTMModel to GRUModel
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # Changed from LSTM to GRU
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # [B,T,N,F] --> [B, T, N*F]

        gru_out, hn = self.gru(x)  # [B,T,N*F] --> [B, T, hidden], changed from LSTM to GRU
        out = self.fc(gru_out)  # [B, T, hidden]  --> [B,T,N*N]
        return out


# The rest of the code remains the same as in the original...

def create_sequences(speed_data, od_data, T):
    inputs = np.expand_dims(speed_data, axis=1)
    targets = np.expand_dims(od_data, axis=1)
    return np.array(inputs), np.array(targets)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    set_seed(42)
    speed = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy')
    od = np.load('../data/OD_完整批处理_3.17_Final_MCM_60.npy')

    mean = 0.05
    std_dev = 0.01
    T, N = speed.shape

    x_random = np.random.normal(loc=mean, scale=std_dev, size=(T, N))
    x_random = np.clip(x_random, 0, 0.1)

    train_size = int(T * 0.6)
    val_size = int(T * 0.2)
    train_size = int(T * 0.9537)
    val_size = int(T * 0.0225)

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    print("6:2:2顺序划分的训练集Speed", speed_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集Speed", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集Speed", speed_test.shape, "OD", od_test.shape)

    od_train_departures = np.sum(od_train, axis=-1)
    mean_speed = np.mean(speed_train, axis=0)
    mean_departures = np.mean(od_train_departures, axis=0)
    temporal = mean_departures - mean_speed

    temporal_expanded_train = np.tile(temporal, (speed_train.shape[0], 1))
    temporal_expanded_val = np.tile(temporal, (speed_val.shape[0], 1))
    temporal_expanded_test = np.tile(temporal, (speed_test.shape[0], 1))

    speed_freq = np.load('../LSTM/data/速度的周期状态_对应25.1.14的速度数据集.npy')
    od_freq = np.load('../LSTM/data/OD的周期状态_对应25.1.14的OD数据集.npy')
    freq = od_freq - speed_freq

    freq_expanded_train = np.tile(freq, (speed_train.shape[0], 1))
    freq_expanded_val = np.tile(freq, (speed_val.shape[0], 1))
    freq_expanded_test = np.tile(freq, (speed_test.shape[0], 1))

    x_train = np.stack([speed_train, temporal_expanded_train, freq_expanded_train, x_random_train], axis=-1)
    x_val = np.stack([speed_val, temporal_expanded_val, freq_expanded_val, x_random_val], axis=-1)
    x_test = np.stack([speed_test, temporal_expanded_test, freq_expanded_test, x_random_test], axis=-1)

    T = 1
    x_train, od_train = create_sequences(x_train, od_train, T)
    x_val, od_val = create_sequences(x_val, od_val, T)
    x_test, od_test = create_sequences(x_test, od_test, T)

    print("训练集 shape:", x_train.shape, "OD形状", od_train.shape)
    print("验证集 shape:", x_val.shape, "OD形状", od_val.shape)
    print("测试集 shape:", x_test.shape, "OD形状", od_test.shape)

    scaler = MinMaxScaler()
    train_3 = x_train[..., :3]
    val_3 = x_val[..., :3]
    test_3 = x_test[..., :3]

    x_train[..., :3] = scaler.fit_transform(train_3.reshape(-1, 3)).reshape(train_3.shape)
    x_val[..., :3] = scaler.transform(val_3.reshape(-1, 3)).reshape(val_3.shape)
    x_test[..., :3] = scaler.transform(test_3.reshape(-1, 3)).reshape(test_3.shape)

    train_data = x_train[...,0]
    val_data = x_val[...,0]
    test_data = x_test[...,0]
    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    print(train_data[0, 0, 66:82, :])

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    print("归一化后的训练集 shape:", train_data.shape, "OD形状", train_target.shape)
    print("归一化后的验证集 shape:", val_data.shape, "OD形状", val_target.shape)
    print("归一化后的测试集 shape:", test_data.shape, "OD形状", test_target.shape)

    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


log_filename = f"log/GRU_MCM完整批处理的调参.log"

def train_model(model, train_loader, val_loader, epochs=100, patience=10, learning_rate=0.001, hidden_size=0, load=0):
    if load == 1:
        model.load_state_dict(torch.load('ckpt/GRU_best_model.pth'))
        logging.info(f"best model loaded")
        print(f"best model loaded")

    learning_rate = learning_rate
    epochs = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), targets.size(1), -1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(targets.size(0), targets.size(1), -1))
                val_loss += loss.item()

        print(
            f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')
        logging.info(
            f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ckpt/GRU_best_model.pth')
            print(f"best model saved at epoch{epoch + 1}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
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


def test_model(model, test_loader, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('ckpt/GRU_best_model.pth'))

    criterion = nn.MSELoss()
    total_rmse = 0.0
    total_mae = 0.0
    test_loss = 0.0
    total_mape = 0
    total_cpc = 0
    total_jsd = 0

    all_real_od = []
    all_pred_od = []

    real_od_sum = torch.zeros(110, 110)
    predicted_od_sum = torch.zeros(110, 110)
    total_samples = 0
    N = 110
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, 110, 110)
            targets = targets.view(-1, 110, 110)
            print(outputs.shape)

            mask = torch.ones_like(targets)
            for i in range(N):
                mask[:, i, i] = 0

            rmse, mae, mape, cpc, jsd = calculate_rmse_mae(outputs * mask, targets)
            total_rmse += rmse
            total_mae += mae
            total_mape += mape
            total_cpc += cpc
            total_jsd += jsd

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    rmse = total_rmse / len(test_loader)
    mae = total_mae / len(test_loader)
    mape = total_mape / len(test_loader)
    cpc = total_cpc / len(test_loader)
    jsd = total_jsd / len(test_loader)
    test_loss = test_loss / len(test_loader)

    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)

    # np.save("../可视化/测试集TNN/Pred_GRU.npy", all_pred_od_t)

    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)

    print(f"Test Loss:{test_loss}")
    print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f} Test MAPE: {mape:.4f} CPC: {cpc:.4f} JSD: {jsd:.4f}')

    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Lr = {learning_rate},Test Loss: {test_loss:.4f} RMSE: {rmse:.4f} MAE: {mae:.4f} MAPE: {mape:.4f} CPC: {cpc:.4f} JSD: {jsd:.4f}\n")


def main():
    lr_list = [0.01, 0.005, 0.001,
               0.0005, 0.0001]
    lr_list = [0.0001]  # 是否当前最佳 YES 6.20

    for lr in lr_list:
        print(f"当前学习率: {lr}")

        train_loader, val_loader, test_loader = load_data()

        model = GRUModel(input_size=110, hidden_size=512, output_size=110 * 110, num_layers=2)  # Changed to GRUModel

        train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=lr, load=0)

        test_model(model, test_loader, learning_rate=lr)


if __name__ == "__main__":
    main()