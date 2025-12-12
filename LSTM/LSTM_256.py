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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # 多层感知机（MLP）层，输入出发量，输出预测的OD矩阵
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, output_size)
        # )

    def forward(self, x):

        x = x.view(x.size(0), x.size(1), -1)  # [B,T,N,F] --> [B, T, N*F]

        lstm_out, (hn, cn) = self.lstm(x)  # [B,T,N*F] --> [B, T, hidden]
        out = self.fc(lstm_out)  # [B, T, hidden]  --> [B,T,N*N]
        # print("FC输出", out.shape)

        return out


# 创建输入和目标数据
def create_sequences(speed_data, od_data, T):
    # inputs, targets = [], []
    # for i in range(len(speed_data) - T + 1):
    #     inputs.append(speed_data[i:i+T])
    #     targets.append(od_data[i:i+T])

    inputs = np.expand_dims(speed_data, axis=1)
    targets = np.expand_dims(od_data, axis=1)
    return np.array(inputs), np.array(targets)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子

def load_data():

    set_seed(42)

    # 加载数据
    data = np.load('../LSTM/data/武汉速度数据集_1KM_110区域_25.1.14.npy')  # 形状 [T, N, 2]
    speed = data[:, :, 0]  # 平均速度 [T, N]
    od = np.load('../LSTM/data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]

    # 设置均值和标准差
    mean = 0.05
    std_dev = 0.01
    # 获取数据长度 T
    T, N = speed.shape

    # 生成T组高斯分布数据
    x_random = np.random.normal(loc=mean, scale=std_dev, size=(T, N))

    # 将数据裁剪到0到1之间
    x_random = np.clip(x_random, 0, 0.1)

    # 按顺序划分数据
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # 输出划分后的数据形状
    print("6:2:2顺序划分的训练集Speed", speed_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集Speed", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集Speed", speed_test.shape, "OD", od_test.shape)

    # 在训练集上计算 OD 出发总量
    od_train_departures = np.sum(od_train, axis=-1)  # 形状 [T_train, N]

    # 计算每个区域在 T_train 时间步的平均速度和平均出发总量
    mean_speed = np.mean(speed_train, axis=0)  # 每个区域的平均速度 [N,]
    mean_departures = np.mean(od_train_departures, axis=0)  # 每个区域的平均出发总量 [N,]

    # 计算平均速度和平均出发总量的差值
    temporal = mean_departures - mean_speed  # [N,]

    # 输出结果
    # print("temporal 变量形状：", temporal.shape)

    # 将 temporal 变量扩展到与输入数据时间维度匹配
    temporal_expanded_train = np.tile(temporal, (speed_train.shape[0], 1))  # [T_train, N]
    temporal_expanded_val = np.tile(temporal, (speed_val.shape[0], 1))  # [T_val, N]
    temporal_expanded_test = np.tile(temporal, (speed_test.shape[0], 1))  # [T_test, N]

    # freq
    speed_freq = np.load('../LSTM/data/速度的周期状态_对应25.1.14的速度数据集.npy')  # 形状 [N,]
    od_freq = np.load('../LSTM/data/OD的周期状态_对应25.1.14的OD数据集.npy')  # 形状 [N,]
    # print(speed_freq.shape)
    # print(od_freq.shape)
    freq = od_freq - speed_freq

    freq_expanded_train = np.tile(freq, (speed_train.shape[0], 1))  # [T_train, N]
    freq_expanded_val = np.tile(freq, (speed_val.shape[0], 1))  # [T_val, N]
    freq_expanded_test = np.tile(freq, (speed_test.shape[0], 1))  # [T_test, N]

    # 添加到训练集、验证集和测试集
    x_train = np.stack([speed_train, temporal_expanded_train, freq_expanded_train, x_random_train],
                       axis=-1)  # [T_train, N, 3]
    x_val = np.stack([speed_val, temporal_expanded_val, freq_expanded_val, x_random_val], axis=-1)  # [T_val, N, 3]
    x_test = np.stack([speed_test, temporal_expanded_test, freq_expanded_test, x_random_test],
                      axis=-1)  # [T_test, N, 3]

    T = 1
    x_train, od_train = create_sequences(x_train, od_train, T)
    x_val, od_val = create_sequences(x_val, od_val, T)
    x_test, od_test = create_sequences(x_test, od_test, T)

    # 打印结果形状
    print("训练集 shape:", x_train.shape, "OD形状", od_train.shape)
    print("验证集 shape:", x_val.shape, "OD形状", od_val.shape)
    print("测试集 shape:", x_test.shape, "OD形状", od_test.shape)



    # 归一化
    scaler = MinMaxScaler()

    train_3 = x_train[...,:3]
    val_3 = x_val[...,:3]
    test_3 = x_test[...,:3]
    
    x_train[..., :3] = scaler.fit_transform(train_3.reshape(-1, 3)).reshape(train_3.shape)
    x_val[...,:3] = scaler.transform(val_3.reshape(-1,3)).reshape(val_3.shape)
    x_test[...,:3] = scaler.transform(test_3.reshape(-1,3)).reshape(test_3.shape)

    train_data = x_train
    val_data = x_val
    test_data = x_test
    
    print(train_data[0,0,66:82,:])

    

    # 4种特征_hidden512_mlp_25年2月17重新调参
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader



def train_model(model, train_loader, val_loader, epochs=100, patience=10,learning_rate=0.001,hidden_size=0,load=0):

    if load == 1 :
        model.load_state_dict(torch.load('ckpt/hidden256/best_model_4种特征_hidden256_25.1.14版本数据集.pth'))
        logging.info(f"best model loaded")
        print(f"best model loaded")

    learning_rate = learning_rate
    epochs = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练过程
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # 5. 训练过程
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Early stopping 参数
    patience = patience  # 允许验证损失没有改善的周期数
    early_stop_counter = 0  # 计数器

    # 4. 设置日志记录
    logging.basicConfig(filename='log/training_4种特征_hidden512_mlp_25年2月17重新调参_25.1.14版本数据集.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.info("-----------------------Starting training-------------------------")
    logging.info(f"learning rate: {learning_rate}")

    # # # 5. 训练过程
    # model.load_state_dict(torch.load('ckpt/best_model_频域.pth'))
    # logging.info(f"best model loaded: {best_val_loss}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Output shape: {outputs.shape}")
            loss = criterion(outputs, targets.view(targets.size(0), targets.size(1), -1))  # 扁平化目标
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(targets.size(0), targets.size(1), -1))  # 扁平化目标
                val_loss += loss.item()

        # if epoch % 20 == 0:
        # 输出训练和验证损失到控制台，并写入日志
        print(
            f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')
        logging.info(
            f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ckpt/hidden256/best_model_4种特征_hidden256_25.1.14版本数据集.pth')
            print(f"best model saved at epoch{epoch+1}")
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1  # 如果没有改善，增加计数器

        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break  # 退出训练

    iters = len(val_losses)

    # 6. 可视化训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iters + 1), train_losses, label='Training Loss')
    plt.plot(range(1, iters + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss with lr={learning_rate}')
    plt.show()



def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))
    return rmse.item(), mae.item()

def test_model(model, test_loader,learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('ckpt/hidden256/best_model_4种特征_hidden256_25.1.14版本数据集.pth'))
    # model.load_state_dict(torch.load('ckpt/hidden256/best_model_4种特征_hidden256_25年2月17重新调参_25.1.14版本数据集_7.4021_1.1971_lr_0.02.pth'))

    # logging.info('load the best model, start testing')

    # 计算 RMSE, MAE，并保存到日志
    criterion = nn.MSELoss()

    total_rmse = 0.0
    total_mae = 0.0
    test_loss = 0.0

    all_real_od = []
    all_pred_od = []

    real_od_sum = torch.zeros(110, 110)  # 初始化真实OD的累计矩阵
    predicted_od_sum = torch.zeros(110, 110)  # 初始化预测OD的累计矩阵
    total_samples = 0  # 累计样本数
    N = 110
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.view(targets.size(0), targets.size(1), -1).to(device)
            outputs = model(inputs)  # B,T,N*N

            outputs = outputs.view(-1, 110, 110)  # 恢复为 [batch*T, 110, 110] 的矩阵
            targets = targets.view(-1, 110, 110)
            print(outputs.shape)

            # 设置对角线掩码
            mask = torch.ones_like(targets)
            for i in range(N):
                mask[:, i, i] = 0  # 对角线上的元素设为 0

            # 计算 RMSE 和 MAE
            rmse = torch.sqrt(((outputs * mask - targets) ** 2).mean()).item()
            mae = (torch.abs(outputs * mask - targets)).mean().item()
            total_rmse += rmse
            total_mae += mae

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

            # 累计真实和预测的 OD 矩阵
            real_od_sum += targets.sum(dim=0)  # 按 batch 累加
            predicted_od_sum += outputs.sum(dim=0)
            total_samples += inputs.size(0)

    # 计算最终的 RMSE 和 MAE
    rmse = total_rmse / len(test_loader)
    mae = total_mae / len(test_loader)
    test_loss = test_loss / len(test_loader)

    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)
    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)

    # 打印结果
    print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}')
    logging.info(f"learning rate: {learning_rate}")
    logging.info(f'Test RMSE: {rmse}, Test MAE: {mae}')
    torch.save(model.state_dict(), f'ckpt/hidden256/best_model_4种特征_hidden256_25.1.14版本数据集_{rmse:.4f}_{mae:.4f}_lr_{learning_rate}.pth')

    T = 1
    # 计算平均真实 OD 和预测 OD
    true_max = all_real_od_t.max()
    pred_max = all_pred_od_t.max()
    print(f"true max: {true_max}, predicted max: {pred_max}")
    T = 1
    # 计算平均真实 OD 和预测 OD
    real_od_avg = real_od_sum / (total_samples * T)
    real_od_avg = real_od_avg.cpu().numpy()

    predicted_od_avg = predicted_od_sum / (total_samples * T)
    predicted_od_avg = predicted_od_avg.cpu().numpy()

    # 绘制线性拟合
    # 计算拟合线
    # slope, intercept, r_value, p_value, std_err = stats.linregress(real_od_avg.flatten(), predicted_od_avg.flatten())
    #
    # # 绘制散点图和拟合线
    # plt.figure(figsize=(8, 6))
    #
    # # 绘制真实样本点（红色）
    # plt.scatter(real_od_avg.flatten(), predicted_od_avg.flatten(), c='red', label='Real OD Average')
    #
    # # 绘制预测样本点（蓝色）
    # plt.scatter(predicted_od_avg.flatten(), real_od_avg.flatten(), c='blue', label='Predicted OD Average')
    #
    # # 绘制拟合线
    # plt.plot(real_od_avg.flatten(), slope * real_od_avg.flatten() + intercept, color='green',
    #          label=f'Fit line: y = {slope:.3f}x + {intercept:.3f}')

    # # 绘制图形
    # plt.xlabel('Real OD Average')
    # plt.ylabel('Predicted OD Average')
    # plt.title('Real vs Predicted OD Average with Fit Line')
    # plt.legend()
    # plt.show()


    style = "Blues"
    # # 输出拟合线方程
    # print(f'Fit Line Equation: y = {slope:.3f}x + {intercept:.3f}')
    np.set_printoptions(precision=2, suppress=True)

    print("-------------------不同时间步上---------------------")
    for i in range(0, all_real_od_t.shape[0],8):
        print("-----真实值------")
        print(np.round(all_real_od_t[i, 60:68, 60:68].astype(np.float32), 1))  # 保留小数点后1位
        print("-----预测值-----")
        print(np.round(all_pred_od_t[i, 60:68, 60:68].astype(np.float32), 1))  # 保留小数点后1位

    print("-------------------平均时间步上---------------------")
    print(real_od_avg[60:68, 60:68].astype(int))  # 保留小数点后1位
    print("----------------------------------------------------")
    print(predicted_od_avg[60:68, 60:68].astype(int))  # 保留小数点后1位

    # print("-------------------平均时间步上---------------------")
    # print(np.round(real_od_avg[60:68, 60:68].astype(np.float32), 1))  # 保留小数点后1位
    # print("----------------------------------------------------")
    # print(np.round(predicted_od_avg[60:68, 60:68].astype(np.float32), 1))  # 保留小数点后1位

    # # 前两个图：展示 0-55 序号的 OD 矩阵的真实和预测值
    # start1, end1 = 0, 56
    # # start1, end1 = 0, 40
    #
    # sub_real_1 = real_od_avg[start1:end1, start1:end1]
    # sub_pred_1 = predicted_od_avg[start1:end1, start1:end1]
    #
    # # 后两个图：展示 56-110 序号的 OD 矩阵的真实和预测值
    # start2, end2 = 56, 111
    # # start2, end2 = 40, 80
    #
    # sub_real_2 = real_od_avg[start2:end2, start2:end2]
    # sub_pred_2 = predicted_od_avg[start2:end2, start2:end2]

    # 统一量纲范围
    vmin = min(real_od_avg.min(), predicted_od_avg.min())
    vmax = max(real_od_avg.max(), predicted_od_avg.max())
    # print(f"real max:{real_od_avg.max()}, real min:{real_od_avg.min()}")
    # print(f"pred max:{predicted_od_avg.max()}, pred min:{predicted_od_avg.min()}")
    # print(f"max: {vmax}, min: {vmin}")

    # 绘制热力图
    plt.figure(figsize=(18, 8))

    # 前两个图展示 0-55 序号的真实和预测 OD 矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(real_od_avg, cmap=style, cbar=True, vmin=vmin, vmax=vmax)
    plt.title('Real OD Heatmap')

    plt.subplot(1, 2, 2)
    sns.heatmap(predicted_od_avg, cmap=style, cbar=True, vmin=vmin, vmax=vmax)
    plt.title('Predicted OD Heatmap')

    plt.tight_layout()
    # plt.show()

    # 绘制热力图
    plt.figure(figsize=(18, 8))

    # 前两个图展示 0-55 序号的真实和预测 OD 矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(real_od_avg[35:68, 35:68], cmap=style, cbar=True, vmin=vmin, vmax=vmax)
    plt.title('Real OD Heatmap')

    plt.subplot(1, 2, 2)
    sns.heatmap(predicted_od_avg[35:68, 35:68], cmap=style, cbar=True, vmin=vmin, vmax=vmax)
    plt.title('Predicted OD Heatmap')

    plt.tight_layout()
    # plt.show()
    print(f"Test Loss:{test_loss}")
    print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}')



def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))
    return rmse.item(), mae.item()



def main():
    train_loader, val_loader, test_loader = load_data()

    model = LSTMModel(input_size=110 * 4, hidden_size=256, output_size=110 * 110, num_layers=2)

    lr = 0.02  # best 0.02
    train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=lr, load=0)

    # 测试模型
    test_model(model, test_loader,learning_rate=lr)


if __name__ == "__main__":
    main()
