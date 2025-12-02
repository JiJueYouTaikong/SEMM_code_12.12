import os
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
from torch.version import cuda

from utils.get_X_Freq import get_X_Freq
import pywt
import torch.nn.functional as F
from utils_v1_4.utils import nb_zeroinflated_nll_loss
import functools
print = functools.partial(print, flush=True)
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
# 新的 ODModel 结构，支持超参：层数、神经元数量、dropout 率
class ODModel(nn.Module):
    def __init__(self, N, temp, freq, num_layers=3, hidden_units=128, dropout_rate=0.1):
        super(ODModel, self).__init__()
        self.N = N
        self.temp = temp
        self.freq = freq

        self.weights = nn.Parameter(torch.randn(N, 3))

        def make_mlp():
            layers = [nn.Linear(N, hidden_units), nn.ReLU(), nn.Dropout(dropout_rate)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU(), nn.Dropout(dropout_rate)]
            layers += [nn.Linear(hidden_units, N * N)]
            return nn.Sequential(*layers)

        self.mlp_n = make_mlp()
        self.mlp_p = make_mlp()
        self.mlp_pi = make_mlp()

    def forward(self, x):
        B, N = x.shape
        x_temp = np.tile(self.temp, (B, 1))
        x_freq = np.tile(self.freq, (B, 1))
        x_random = np.clip(np.random.normal(0.05, 0.01, size=(B, N)), 0, 0.1)
        x_temp_freq_rand = np.stack([x_temp, x_freq, x_random], axis=-1)
        tensor_delta_x = torch.tensor(x_temp_freq_rand, dtype=torch.float, device=x.device)

        self.weights = self.weights.to(x.device)
        weighted_sum = x + torch.sum(tensor_delta_x * self.weights.unsqueeze(0), dim=2)

        n_flat = self.mlp_n(weighted_sum)
        p_flat = self.mlp_p(weighted_sum)
        pi_flat = self.mlp_pi(weighted_sum)

        n = F.softplus(n_flat.view(B, N, N))
        p = torch.sigmoid(p_flat.view(B, N, N))
        pi = torch.sigmoid(pi_flat.view(B, N, N))
        return n, p, pi



# 训练过程
def train_model(model, train_loader, val_loader, epochs=100, patience=10, learning_rate=0.001, is_mcm=None, freq_method=None,stop_type=None):

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    best_val_loss = float('inf')
    patience_counter = 0
    # 训练过程
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()
            train_n, train_p, train_pi = model(inputs)
            loss = nb_zeroinflated_nll_loss(targets, train_n, train_p, train_pi)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算训练集的平均损失
        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                n_val,p_val,pi_val = model(inputs)

                loss = nb_zeroinflated_nll_loss(targets, n_val,p_val,pi_val)
                val_loss += loss.item()

                val_pred = (1 - pi_val.detach().cpu().numpy()) * (
                            n_val.detach().cpu().numpy() / p_val.detach().cpu().numpy() - n_val.detach().cpu().numpy())

                mae = np.mean(np.abs(val_pred - targets.detach().cpu().numpy()))
                val_mae += mae.item()

                mse = np.mean((val_pred - targets.detach().cpu().numpy()) ** 2)
                val_rmse += np.sqrt(mse)

        # 计算验证集的平均损失
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_rmse /= len(val_loader)

        if epoch % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train NLL Loss: {train_loss:.4f}, Val NLL Loss: {val_loss:.4f}, Val Pred MAE: {val_mae:.4f} , Val RMSE: {val_rmse:.4f}")


        if stop_type == 'rmse':
            loss_value = val_rmse
        elif stop_type == 'nll':
            loss_value = val_loss
        else:
            return -1

        # 提前停止机制
        if loss_value < best_val_loss:
            best_val_loss = loss_value

            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"ckpt/v1_4_data_shuf/best_model_{freq_method}_is_mcm_{is_mcm}.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


    return best_val_loss


def evaluate_metrics(predictions, targets):
    '''
    单个方法的五项指标计算：RMSE, MAE, MAPE, CPC, JSD
    '''
    predictions = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    # MAPE
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
    for t in range(pred_flat.shape[0]):
        pred_t = pred_flat[t]
        targ_t = targ_flat[t]

        epsilon = 1e-8
        pred_dist = (pred_t + epsilon) / (torch.sum(pred_t) + epsilon * pred_t.numel())
        targ_dist = (targ_t + epsilon) / (torch.sum(targ_t) + epsilon * targ_t.numel())

        m = 0.5 * (pred_dist + targ_dist)
        kl1 = torch.sum(pred_dist * torch.log(pred_dist / m))
        kl2 = torch.sum(targ_dist * torch.log(targ_dist / m))
        jsd_t = 0.5 * (kl1 + kl2)
        jsd_list.append(jsd_t.item())
    jsd = sum(jsd_list) / len(jsd_list)

    return rmse.item(), mae.item(), mape.item(), cpc, jsd

def calculate_rmse_mae(predictions, targets):
    '''

    :param predictions: [T,N,N] T个时间步上的OD矩阵预测值
    :param targets:  [T,N,N] T个时间步上的OD矩阵真值
    :return:
    '''
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
def load_data(is_mcm=False,freq_shuffle=True,freq_method=None,data_shuf=None):

    log_filename = f"log/v1_4/{freq_method}_MCM_{is_mcm}_data_shuf_{data_shuf}_freq_shuf_{freq_shuffle}.log"
    set_seed(42)

    if is_mcm:
        speed = np.load('data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('data/OD_完整批处理_3.17_Final_MCM_60.npy')


        # 获取数据长度 T
        T, N = speed.shape

        # 根据指定的时间步数划分
        test_size = 35
        val_size = 33
        train_size = T - test_size - val_size

    else:
        speed = np.load('data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('data/OD_完整批处理_3.17_Final.npy')

        # 获取数据长度 T
        T, N = speed.shape

        # 按顺序划分数据
        train_size = int(T * 0.6)
        val_size = int(T * 0.2)

    # 构建索引
    all_indices = np.arange(T)
    if data_shuf:
        print("[INFO] ************************** data shuffle! *************************************")
        np.random.seed(42)  # 固定随机种子以确保结果可复现
        np.random.shuffle(all_indices)

    # 按索引划分
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]


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

    batch = 32
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    scaler = MinMaxScaler()
    x_temp = scaler.fit_transform(temporal.reshape(-1, 1))

    scaler = MinMaxScaler()
    x_freq = scaler.fit_transform(freq.detach().numpy().reshape(-1, 1))

    x_temp = np.squeeze(x_temp, axis=-1)
    x_freq = np.squeeze(x_freq, axis=-1)

    return train_loader, val_loader, test_loader, x_temp, x_freq,log_filename


# 测试
def test_model(model, test_loader, lr: float,log_name=None,patience=30, is_mcm=None, freq_method=None,stop_type=None):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    path = f"ckpt/v1_4_data_shuf/best_model_{freq_method}_is_mcm_{is_mcm}.pth"

    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("已加载模型参数")
    else:
        print(f"模型参数文件不存在：{path}")

    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0
    cpc_total = 0
    jsd_total = 0
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


            # 推理输出
            n_test, p_test, pi_test = model(inputs)

            # 计算loss
            loss = nb_zeroinflated_nll_loss(targets, n_test, p_test, pi_test)
            test_loss += loss.item()

            # 计算预测值
            mean_pred = (1 - pi_test.detach().cpu().numpy()) * (n_test.detach().cpu().numpy() / p_test.detach().cpu().numpy() - n_test.detach().cpu().numpy())
            mean_pred = torch.tensor(mean_pred, dtype=torch.float32).to(device)

            # 计算 RMSE 和 MAE
            rmse, mae, mape,cpc,jsd = calculate_rmse_mae(mean_pred * mask, targets)
            rmse_total += rmse
            mae_total += mae
            mape_total += mape
            cpc_total += cpc
            jsd_total += jsd

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(mean_pred.cpu() * mask.cpu().numpy())

    print(f"测试集len={len(test_loader)}")
    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)
    cpc_total /= len(test_loader)
    jsd_total /= len(test_loader)

    print(f"Test NLL Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f}  MAE: {mae_total:.4f} MAPE: {mape_total:.4f} CPC: {cpc_total:.4f} JSD: {jsd_total:.4f}")


    # 计算平均的OD矩阵
    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)


    print(f"Device:{device}")
    with open(log_name, 'a') as log_file:
        log_file.write(f"Patience={patience} Loss: {test_loss:.4f} 【RMSE】: 【{rmse_total:.4f}】 MAE: {mae_total:.4f} MAPE: {mape_total:.4f} CPC: {cpc_total:.4f} JSD: {jsd_total:.4f}\n")

    # 删除微调保存的最优模型权重
    path = f"ckpt/v1_4_data_shuf/best_model_{freq_method}_is_mcm_{is_mcm}.pth"
    if os.path.exists(path):
        os.remove(path)

# 主程序
# 新 main 函数，支持多层嵌套超参搜索
from itertools import product


def main():
    is_mcm_list = [True, False]
    freq_methods = ['WT', 'STFT']
    stop_types = ['rmse', 'nll']
    num_layers_list = [2, 3, 4, 5, 6, 8, 10]
    hidden_units_list = [64, 128, 256, 512]
    dropout_list = [0.0, 0.1, 0.2]
    lr_list = [0.05, 0.045, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005,
               0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001,
               0.0005, 0.0004]
    patience = 20
    data_shuf = True

    for is_mcm, freq_method, stop_type in product(is_mcm_list, freq_methods, stop_types):


        for num_layers, hidden_units, dropout_rate in product(num_layers_list, hidden_units_list, dropout_list):
            for lr in lr_list:
                torch.cuda.empty_cache()

                train_loader, val_loader, test_loader, temp, freq, log_filename = load_data(
                    is_mcm=is_mcm, freq_shuffle=False, freq_method=freq_method,data_shuf=data_shuf)

                param = f"MCM={is_mcm}, Method={freq_method}, Stop={stop_type}, Layers={num_layers}, Hidden={hidden_units}, Dropout={dropout_rate}, LR={lr}"
                print(param)

                model = ODModel(
                    N=110,
                    temp=temp,
                    freq=freq,
                    num_layers=num_layers,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate
                )

                train_model(
                    model, train_loader, val_loader,
                    epochs=2000, patience=patience,
                    learning_rate=lr,
                    is_mcm=is_mcm,
                    freq_method=freq_method,
                    stop_type=stop_type
                )

                with open(log_filename, 'a') as log_file:
                    log_file.write(f"\t \t (Param: {param})\n")


                test_model(
                    model, test_loader,
                    lr=lr,
                    log_name=log_filename,
                    patience=patience,
                    is_mcm=is_mcm,
                    freq_method=freq_method,
                    stop_type=stop_type
                )


if __name__ == "__main__":
    main()
