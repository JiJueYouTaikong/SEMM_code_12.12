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

# 数据准备
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val





# 定义模型
class ODModel(nn.Module):
    def __init__(self, N):
        super(ODModel, self).__init__()
        # 线性层，拟合每个区域的出发量
        # self.linear = nn.Linear(4*N, 1*N)  # 输入4个特征，输出1个出发量
        self.linear = nn.Linear(4, 1)  # 输入4个特征，输出1个出发量

        # 多层感知机（MLP）层，输入出发量，输出预测的OD矩阵
        self.mlp = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N * N)
        )
        self.N = N

    def forward(self, x):
        # 输入 x 的形状：[batch, N, 4]
        batch_size, N, _ = x.shape

        # x = x.view(batch_size,-1)

        # 线性拟合出每个区域的出发量，形状：[batch, N]
        out = self.linear(x)  # 形状：[batch, N, 1]

        print("线性层输出", out.shape)
        departure_volume = out.squeeze(dim=-1)  # 形状：[batch, N]

        # 输入到MLP网络中
        od_matrix_flat = self.mlp(departure_volume)  # 形状：[batch, N * N]

        # 重塑输出为OD矩阵的形状：[batch, N, N]
        od_matrix = od_matrix_flat.view(batch_size, N, N)

        return od_matrix

# 保存日志文件
log_filename = f"log/training_log_25.1.14版本_{time.strftime('%Y%m%d_%H%M%S')}.log"
with open(log_filename, 'w') as log_file:
    log_file.write("Epoch, Train Loss, Validation Loss\n")

# 训练过程
def train_model(model, train_loader, val_loader, epochs=100, patience=10,learning_rate=0.001, load=0):


    if load == 1 :
        # model.load_state_dict(torch.load('best_ckpt/best_model_feature4_49.8672_6.0555.pth'))
        model.load_state_dict(torch.load('ckpt/best_model_feature4_25.1.14数据集版本.pth'))

        print(f"best model loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0



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

        # 保存每一轮的损失，并打印
        with open(log_filename, 'a') as log_file:
            log_file.write(f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}\n")

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/best_model_feature4_25.1.14数据集版本.pth")
            print(f"best model saved at epoch{epoch + 1},当前最好验证集误差：{best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 绘制训练过程中的损失曲线
    with open(log_filename, 'r') as log_file:
        epochs_list, train_loss_list, val_loss_list = [], [], []
        for line in log_file.readlines()[1:]:
            epoch, train_loss, val_loss = line.strip().split(", ")
            epochs_list.append(int(epoch))
            train_loss_list.append(float(train_loss))
            val_loss_list.append(float(val_loss))

    plt.plot(epochs_list, train_loss_list, label='Train Loss')
    plt.plot(epochs_list, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig("loss_curve.png")
    plt.show()



def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))
    return rmse.item(), mae.item()


# 加载数据
def load_data():

    # 加载数据
    data = np.load('data/武汉速度数据集_1KM_110区域_25.1.14.npy')  # 形状 [T, N, 2]
    speed = data[:, :, 0]  # 平均速度 [T, N]
    od = np.load('data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]

    # 设置随机种子
    np.random.seed(42)  # 42 是固定的种子，可以改成其他值

    # 设置均值和标准差
    mean = 0.05
    std_dev = 0.01
    # 获取数据长度 T
    T, N = speed.shape

    # 生成T组高斯分布数据
    x_random = np.random.normal(loc=mean, scale=std_dev, size=(T, N))

    # 将数据裁剪到0到1之间
    x_random = np.clip(x_random, 0, 0.1)

    # 随机打乱索引
    indices = np.arange(T)
    np.random.seed(42)  # 固定随机种子，确保可复现
    np.random.shuffle(indices)

    # 按6:2:2划分索引
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # 输出划分后的数据形状
    print("训练集：速度形状", speed_train.shape, "OD形状", od_train.shape)
    print("验证集：速度形状", speed_val.shape, "OD形状", od_val.shape)
    print("测试集：速度形状", speed_test.shape, "OD形状", od_test.shape)

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
    speed_freq = np.load('data/速度的周期状态_对应25.1.14的速度数据集.npy')  # 形状 [N,]
    od_freq = np.load('data/OD的周期状态_对应25.1.14的OD数据集.npy')  # 形状 [N,]
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

    # 打印结果形状
    print("训练集 shape:", x_train.shape, "OD形状", od_train.shape)
    print("验证集 shape:", x_val.shape, "OD形状", od_val.shape)
    print("测试集 shape:", x_test.shape, "OD形状", od_test.shape)



    # 硬编码归一化
    # scaler = MinMaxScaler()
    # train_data = scaler.fit_transform(x_train.reshape(-1, 4)).reshape(100, 110, 4)
    # val_data = scaler.transform(x_val.reshape(-1, 4)).reshape(33, 110, 4)
    # test_data = scaler.transform(x_test.reshape(-1, 4)).reshape(35, 110, 4)

    # 归一化
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(x_train.reshape(-1, 4)).reshape(x_train.shape)
    val_data = scaler.transform(x_val.reshape(-1, 4)).reshape(x_val.shape)
    test_data = scaler.transform(x_test.reshape(-1, 4)).reshape(x_test.shape)


    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("训练集 shape:", train_data.shape, "OD形状", train_target.shape)
    print("验证集 shape:", val_data.shape, "OD形状",val_target.shape)
    print("测试集 shape:", test_data.shape, "OD形状", test_target.shape)


    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


# 测试
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_model_feature4_25.1.14数据集版本.pth"))
    # model.load_state_dict(torch.load("best_ckpt/best_model_feature4_49.27_6.0909.pth"))

    # 打印参数
    for name, param in model.named_parameters():
        print(f'{name} shape: {param.shape}')


    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    criterion = nn.MSELoss()

    all_real_od = []
    all_pred_od = []

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            print(f"输入:{inputs.shape},标签:{targets.shape}")

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 计算 RMSE 和 MAE
            rmse, mae = calculate_rmse_mae(outputs, targets)
            rmse_total += rmse
            mae_total += mae

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f}")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Test RMSE: {rmse_total:.4f}, MAE: {mae_total:.4f}\n")
    torch.save(model.state_dict(), f"ckpt/best_model_feature4_25.1.14数据集版本_{rmse_total:.4f}_{mae_total:.4f}.pth")

    # 计算平均的OD矩阵
    all_real_od = np.mean(np.concatenate(all_real_od, axis=0), axis=0)
    all_pred_od = np.mean(np.concatenate(all_pred_od, axis=0), axis=0)


    print(all_real_od[60:68, 60:68].astype(int))
    print("--------------------------------------------------")
    print(all_pred_od[60:68, 60:68].astype(int))

    vmin = min(all_pred_od.min(), all_real_od.min())
    vmax = max(all_pred_od.max(), all_real_od.max())
    print(f"real max:{all_real_od.max()}, real min:{all_real_od.min()}")
    print(f"pred max:{all_pred_od.max()}, pred min:{all_pred_od.min()}")
    print(f"Max: {vmax:.4f}, Min: {vmin:.4f}")



    plt.figure(figsize=(15, 7))

    # 绘制真实 OD 热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(all_real_od, cmap="Blues", cbar=True,vmin=vmin, vmax=vmax)
    plt.title("Average True OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 绘制预测 OD 热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(all_pred_od, cmap="Blues", cbar=True,vmin=vmin, vmax=vmax)
    plt.title("Average Predicted OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 7))

    # 绘制真实 OD 热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(all_real_od[35:68, 35:68], cmap="Blues", cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average True OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 绘制预测 OD 热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(all_pred_od[35:68, 35:68], cmap="Blues", cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average Predicted OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


# 主程序
def main():
    train_loader, val_loader, test_loader = load_data()
    model = ODModel(N=110)  # N为区域数

    # 训练模型
    # train_model(model, train_loader, val_loader, epochs=1000, patience=20, learning_rate=0.000005, load=1)

    # 测试模型
    test_model(model, test_loader)


if __name__ == "__main__":
    main()