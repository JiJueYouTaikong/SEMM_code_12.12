import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import stft
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.get_X_Freq import get_X_Freq
import pywt


# 数据准备
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val


class STFTMLP(nn.Module):
    def __init__(self, T, fs=1.0, nperseg=64, noverlap=32, hidden_size=128):
        super(STFTMLP, self).__init__()
        self.T = T
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

        # 计算STFT后的频率和时间点数
        _, _, Zxx_example = stft(np.zeros(T), fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        n_frequencies, n_times = Zxx_example.shape

        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_frequencies * n_times, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, T)
        )

        # He初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, x1, x2):
        N = x1.shape[0]
        amplitudes_diff = []

        for i in range(N):
            # 对每个样本应用STFT
            f1, t_stft1, Zxx1 = stft(x1[i], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            f2, t_stft2, Zxx2 = stft(x2[i], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)

            # 提取振幅
            amplitude1 = np.abs(Zxx1)
            amplitude2 = np.abs(Zxx2)

            # 计算振幅差值
            diff = amplitude1 - amplitude2
            amplitudes_diff.append(diff)

            # if i == 77:
            #     print(f"speed of {i}:{x1[i][:-20]}")
            #     print(f"speed length:{x1[i].shape}")
            #     print(f"STFT time steps (in hours): {t_stft1}")
            #     print(f"STFT time delta: {np.diff(t_stft1)}")
            #     print(f"amplitudes shape: {amplitude1.shape}")
            #
            #     np.save('可视化/频域/amplitude1.npy', amplitude1)
            #     np.save('可视化/频域/amplitude2.npy', amplitude2)
            #     np.save('可视化/频域/amplitudes_diff.npy', amplitudes_diff)
            #     np.save('可视化/频域/t_stft1.npy', t_stft1)
            #     np.save('可视化/频域/t_stft2.npy', t_stft2)
            #     np.save('可视化/频域/f1.npy', f1)
            #     np.save('可视化/频域/f2.npy', f2)

        # 将差值转换为张量
        amplitudes_diff = np.array(amplitudes_diff)
        amplitudes_diff = torch.tensor(amplitudes_diff, dtype=torch.float32)

        # 通过MLP
        output = self.mlp(amplitudes_diff)  # [N,T]

        output = torch.mean(output, dim=-1)  # [N,]

        return output


class WaveletMLP(nn.Module):
    def __init__(self, T, fs=1.0, wavelet='morl', scales=np.arange(1, 64), hidden_size=128):
        super(WaveletMLP, self).__init__()
        self.T = T
        self.fs = fs
        self.wavelet = wavelet
        self.scales = scales

        # 计算小波变换后的特征维度
        coef_example, _ = pywt.cwt(np.zeros(T), scales, wavelet, sampling_period=1 / fs)
        n_scales, _ = coef_example.shape

        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_scales * T, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, T)
        )

        # He初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        N = x1.shape[0]
        wavelet_diff = []

        for i in range(N):
            # 对每个样本应用小波变换
            coef1, _ = pywt.cwt(x1[i].cpu().numpy(), self.scales, self.wavelet, sampling_period=1 / self.fs)
            coef2, _ = pywt.cwt(x2[i].cpu().numpy(), self.scales, self.wavelet, sampling_period=1 / self.fs)

            # 提取振幅
            amplitude1 = np.abs(coef1)
            amplitude2 = np.abs(coef2)

            # 计算振幅差值
            diff = amplitude1 - amplitude2
            wavelet_diff.append(diff)

        # 将差值转换为张量
        wavelet_diff = np.array(wavelet_diff)
        wavelet_diff = torch.tensor(wavelet_diff, dtype=torch.float32, device=x1.device)

        # 通过MLP
        output = self.mlp(wavelet_diff)  # [N,T]

        output = torch.mean(output, dim=-1)  # [N,]

        return output


class FFTMLP(nn.Module):
    def __init__(self, T, fs=1.0, hidden_size=128):
        super(FFTMLP, self).__init__()
        self.T = T
        self.fs = fs

        # FFT后特征维度（只取正频率部分）
        n_frequencies = T // 2 + 1

        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_frequencies * 2, hidden_size),  # *2 是因为同时包含实部和虚部
            nn.ReLU(),
            nn.Linear(hidden_size, T)
        )

        # He初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        N = x1.shape[0]
        fft_diff_list = []

        # 计算FFT并处理每个样本
        for i in range(N):
            # 使用PyTorch的rfft计算实值信号的FFT
            fft1 = torch.fft.rfft(x1[i], dim=-1)
            fft2 = torch.fft.rfft(x2[i], dim=-1)

            # 计算FFT差值（结合实部和虚部）
            diff = fft1 - fft2

            # 将复数转换为实值特征（实部和虚部）
            real_part = diff.real
            imag_part = diff.imag

            # 合并实部和虚部特征
            fft_features = torch.cat([real_part, imag_part], dim=-1)
            fft_diff_list.append(fft_features)

        # 合并所有样本的特征
        fft_diff = torch.stack(fft_diff_list, dim=0)

        # 通过MLP
        output = self.mlp(fft_diff)  # [N,T]

        output = torch.mean(output, dim=-1)  # [N,]

        return output

# 定义模型
class ODModel(nn.Module):
    def __init__(self, N, temp, freq):
        super(ODModel, self).__init__()

        self.N = N  # 区域数
        self.temp = temp  # 频域差距 [N,]
        self.freq = freq  # 频域差距 [N,]
        n1 = 128  # 隐藏层神经元1
        n2 = 64  # 隐藏层神经元2

        # 权重[α1,α2,α3] --> shape [N, 3]
        self.weights = nn.Parameter(torch.randn(N, 3))

        # MLP层 input [B,N] --> output [B,N*N]
        self.mlp = nn.Sequential(
            nn.Linear(N, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, N * N)
        )

    def forward(self, x):
        '''
        前向传播
        :param x: Batch输入Speed --> [B,N]
        :return: Batch输出OD     --> [B,N,N]
        '''
        B,N = x.shape  # [B,N]

        x_temp = np.tile(self.temp, (B, 1))  # [B,N]
        x_freq = np.tile(self.freq, (B, 1))  # [B,N]

        x_random = np.random.normal(loc=0.05, scale=0.01, size=(B, N)).astype(float)  # [B,N]
        x_random = np.clip(x_random, 0, 0.1)

        x_temp_freq_rand = np.stack([x_temp, x_freq, x_random], axis=-1)  # [B,N,3]
        tensor_delta_x = torch.tensor(x_temp_freq_rand,dtype=torch.float).to(x.device)

        self.weights = self.weights.to(x.device)
        weighted_sum = x + torch.sum(tensor_delta_x * self.weights.unsqueeze(0), dim=2)


        # 输入到MLP网络中
        od_matrix_flat = self.mlp(weighted_sum)  # [B,N] --> [B, N * N]
        # reshape 最终输出
        od_matrix = od_matrix_flat.view(B, N, N)  # [B, N, N]

        return od_matrix


# 训练过程
def train_model(model, train_loader, val_loader, epochs=100, patience=10, learning_rate=0.001, load=0):
    if load == 1:
        model.load_state_dict(torch.load('ckpt/v1_3/best_model_feature4_25.1.14数据集版本.pth'))
        print(f"best model loaded")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    N = 110

    # 训练过程
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

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
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # 计算验证集的平均损失
        val_loss /= len(val_loader)


        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/v1_3/best_model_feature4_25.1.14数据集版本.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


    return best_val_loss

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


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子


# 加载数据
def load_data(is_mcm=False,freq_shuffle=True,freq_method=None):

    log_filename = f"log/v1_3/A完整批处理的调参_{freq_method}_MCM_{is_mcm}_Shuffle_{freq_shuffle}.log"
    set_seed(42)

    # 加载数据
    # data = np.load('data/武汉速度数据集_1KM_110区域_25.1.14.npy')  # 形状 [T, N, 2]
    # speed = data[:, :, 0]  # 平均速度 [T, N]
    #
    # od = np.load('data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]
    if is_mcm:
        speed = np.load('data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('data/OD_完整批处理_3.17_Final_MCM_60.npy')

        # 获取数据长度 T
        T, N = speed.shape

        train_size = int(T * 0.9537)
        val_size = int(T * 0.0225)

    else:
        speed = np.load('data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('data/OD_完整批处理_3.17_Final.npy')

        # 获取数据长度 T
        T, N = speed.shape

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
    # x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

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
    temporal = (mean_departures - mean_speed).astype(float)  # [N,]


    # 频域差距求解
    seq1 = torch.tensor(speed_train, dtype=torch.float32).transpose(0, 1)  # [N,T]
    seq2 = torch.tensor(od_train_departures, dtype=torch.float32).transpose(0, 1) # [N,T]

    if freq_shuffle:
        # 生成打乱索引
        perm = torch.randperm(seq1.size(1))
        # 按照相同的索引打乱两个序列
        seq1 = seq1[:, perm]  # [N, T]
        seq2 = seq2[:, perm]  # [N, T]

    if freq_method == 'STFT':
        stftmlp = STFTMLP(train_size)
        freq = stftmlp(seq1, seq2)
    elif freq_method == 'WT':
        wtmlp = WaveletMLP(train_size)
        freq = wtmlp(seq1, seq2)
    elif freq_method == 'FFT':
        fftmlp = FFTMLP(train_size)
        freq = fftmlp(seq1, seq2)



    # 归一化
    scaler = MinMaxScaler()
    # print(f"test_data: {speed_test[-2, 10:20]}")
    train_data = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    val_data = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    test_data = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)

    # print(f"train_data: {train_data[:10, 66]}")

    # print(f"test_data: {test_data[-2, 10:20]}")

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("归一化后的训练集 shape:", train_data.shape, "OD形状", train_target.shape)
    print("归一化后的验证集 shape:", val_data.shape, "OD形状", val_target.shape)
    print("归一化后的测试集 shape:", test_data.shape, "OD形状", test_target.shape)

    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    scaler = MinMaxScaler()
    x_temp = scaler.fit_transform(temporal.reshape(-1, 1))

    scaler = MinMaxScaler()
    x_freq = scaler.fit_transform(freq.detach().numpy().reshape(-1, 1))

    x_temp = np.squeeze(x_temp, axis=-1)
    x_freq = np.squeeze(x_freq, axis=-1)

    return train_loader, val_loader, test_loader, x_temp, x_freq,log_filename


# 测试
def test_model(model, test_loader, lr: float,log_name=None,patience=30):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/v1_3/best_model_feature4_25.1.14数据集版本.pth"))

    # 打印结构
    # summary(model, input_size=(32, 110, 4))

    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0
    criterion = nn.MSELoss()
    N = 110

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
            rmse, mae, mape = calculate_rmse_mae(outputs * mask, targets)
            rmse_total += rmse
            mae_total += mae
            mape_total += mape

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f}")

    torch.save(model.state_dict(),
               f"ckpt/v1_3/best_model_feature4_25.1.14数据集版本_{test_loss:.4f}_{rmse_total:.4f}_{mae_total:.4f}_lr_{lr}.pth")

    # 计算平均的OD矩阵
    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)

    np.save("./可视化/测试集TNN/Pred_ours_WT_Shuffle.npy", all_pred_od_t)
    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)

    print("-------------------不同时间步上---------------------")
    for i in range(0, all_real_od_t.shape[0]):
        maxi = all_real_od_t[i].max()
        mini = all_real_od_t[i].min()
        maxi_pred = all_pred_od_t[i].max()
        mini_pred = all_pred_od_t[i].min()
        print(f"-----真实值 step{i} {maxi:.2f} {mini:.2f}------")
        print(np.round(all_real_od_t[i, 60:68, 60:68].astype(np.float32)))  # 保留小数点后1位
        print(f"-----预测值 step{i} {maxi_pred:.2f} {mini_pred:.2f}------")
        print(np.round(all_pred_od_t[i, 60:68, 60:68].astype(np.float32)))  # 保留小数点后1位

    print("-------------------平均时间步上---------------------")
    print(np.round(all_real_od[60:68, 60:68].astype(np.float32), 0).astype(int))  # 保留小数点后1位
    print("----------------------------------------------------")
    print(np.round(all_pred_od[60:68, 60:68].astype(np.float32), 0).astype(int))  # 保留小数点后1位

    vmin = min(all_pred_od.min(), all_real_od.min())
    vmax = max(all_pred_od.max(), all_real_od.max())

    true_max = all_real_od_t.max()
    pred_max = all_pred_od_t.max()
    # print(f"真实最大值{true_max},预测最大值，{pred_max}")
    

   #  colors = "Blues"  # Blues YlGn Greens YlGnBu
   #
   #
   #  plt.figure(figsize=(15, 7))
   #
   #  # 绘制真实 OD 热力图
   #  plt.subplot(1, 2, 1)
   #  sns.heatmap(all_real_od, cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
   #  plt.title("Average True OD Matrix", fontsize=14)
   #  plt.xlabel("Destination Zones")
   #  plt.ylabel("Origin Zones")
   #
   #  # 绘制预测 OD 热力图
   #  plt.subplot(1, 2, 2)
   #  sns.heatmap(all_pred_od, cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
   #  plt.title("Average Predicted OD Matrix", fontsize=14)
   #  plt.xlabel("Destination Zones")
   #  plt.ylabel("Origin Zones")
   #
   #  # 调整布局并显示
   #  plt.tight_layout()
   #  # plt.show()
   #  plt.savefig("ODE1.3_1.png")
   #
   #
   #  start_id = 30  # 35
   #  end_id = 70  # 68
   #  vmin = min(all_pred_od[start_id:end_id, start_id:end_id].min(), all_real_od[start_id:end_id, start_id:end_id].min())
   #  vmax = max(all_pred_od[start_id:end_id, start_id:end_id].max(), all_real_od[start_id:end_id, start_id:end_id].max())
   #
   #  plt.figure(figsize=(15, 7))
   #  # 绘制真实 OD 热力图
   #  plt.subplot(1, 2, 1)
   #  sns.heatmap(all_real_od[start_id:end_id, start_id:end_id], cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
   #  plt.title("Average True OD Matrix", fontsize=14)
   #  plt.xlabel("Destination Zones")
   #  plt.ylabel("Origin Zones")
   #
   #  # 绘制预测 OD 热力图
   #  plt.subplot(1, 2, 2)
   #  sns.heatmap(all_pred_od[start_id:end_id, start_id:end_id], cmap=colors, cbar=True, vmin=vmin, vmax=vmax)
   #  plt.title("Average Predicted OD Matrix", fontsize=14)
   #  plt.xlabel("Destination Zones")
   #  plt.ylabel("Origin Zones")
   #
   # #  调整布局并显示
   #  plt.tight_layout()
   #  plt.savefig("ODE1.3_2.png")
   #  # plt.show()
   #
   #
   #
   #  # 绘制散点图
   #  all_real_od_flat = all_real_od_t.flatten()
   #  all_pred_od_flat = all_pred_od_t.flatten()
   #
   #  plt.figure(figsize=(9, 8))
   #  plt.scatter(all_real_od_flat, all_pred_od_flat, alpha=0.5)
   #
   #  # 设置坐标轴范围从 0 开始
   #  max_val = np.max([np.max(all_real_od_flat), np.max(all_pred_od_flat)])
   #  plt.xlim(0, max_val)
   #  plt.ylim(0, max_val)
   #
   #  # 绘制对角线
   #  lims = [0, max_val]
   #  plt.plot(lims, lims, color='red', linewidth=2, linestyle='-', alpha=0.7)
   #
   #  plt.xlabel('True OD')
   #  plt.ylabel('Pred OD')
   #  plt.title('The relationship between real and predicted OD')
   #  plt.savefig('ODE1.3_3.png')
   #  # plt.show()





    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f}")
    print(f"Device:{device}")
    with open(log_name, 'a') as log_file:
        log_file.write(f"Lr = {lr},patience = {patience},Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f}\n")


# 主程序
def main():



    # 定义学习率列表
    lr_list = [0.1, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005,
               0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001,
               0.0008, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
    # lr_list = [0.03]  # 20 0.03
    lr_list = [0.0035]  # WT MCM best patience20
    lr_list =[0.02] # STFT MCM not shuffle best
    lr_list = [0.03]  # STFT not MCM not shuffle best
    patience = 20

    # 遍历学习率列表
    for lr in lr_list:
        print(f"当前学习率: {lr}")


        train_loader, val_loader, test_loader, temp, freq, log_filename = load_data(is_mcm=False,freq_shuffle=False,freq_method='STFT')

        model = ODModel(N=110, temp=temp, freq=freq)  # N为区域数

        # 训练模型
        train_model(model, train_loader, val_loader, epochs=1000, patience=patience, learning_rate=lr, load=0)

        # 测试模型
        test_model(model, test_loader, lr=lr,log_name=log_filename,patience=patience)


if __name__ == "__main__":
    main()
