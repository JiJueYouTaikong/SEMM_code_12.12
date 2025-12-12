#%% md
# ## 导入第三方库
#%%
# 导入第三方库
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
#%% md
# ## 加载指定路径的速度和OD数据集
#%%
# 设置数据集路径
# speed_path = 'data/武汉速度数据集_1KM_110区域_25.1.14_TN.npy'
# od_path = 'data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy'
speed_path = 'data/Speed_完整批处理_3.17_Final.npy'
od_path = 'data/OD_完整批处理_3.17_Final.npy'
seed = 42
BATCH = 32

# Load_data函数
def load_data(speed_path, od_path, seed, BATCH):
    # 设置随机种子
    set_seed(seed)

    # 加载数据
    speed = np.load(speed_path)  # speed --> 形状 [T, N]

    od = np.load(od_path)  # OD --> 形状 [T, N, N]

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


    # 在训练集上计算 OD 出发总量
    od_train_production = np.sum(od_train, axis=-1)  # 形状 [T_train, N]

    # 计算每个区域在 T_train 时间步的平均速度和平均出发总量
    mean_speed = np.mean(speed_train, axis=0)  # 每个区域的平均速度 [N,]
    mean_production = np.mean(od_train_production, axis=0)  # 每个区域的平均出发总量 [N,]
    
    # 计算时域差距
    temporal = (mean_production - mean_speed).astype(float)  # [N,]

    # 计算频域差距
    speed_freq = get_X_Freq(speed)  # 形状 [N,]
    od_production = np.sum(od, axis=-1)
    od_freq = get_X_Freq(od_production)  # 形状 [N,]
    freq = (od_freq - speed_freq).astype(float)

    # 频域差距、时域差距、speed归一化
    scaler = MinMaxScaler()
    x_temp = scaler.fit_transform(temporal.reshape(-1, 1))
    x_temp = np.squeeze(x_temp, axis=-1)  # [N,]
    
    scaler = MinMaxScaler()
    x_freq = scaler.fit_transform(freq.reshape(-1, 1))
    x_freq = np.squeeze(x_freq, axis=-1)  # [N,]
    
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    val_data = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    test_data = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)


    # tensor张量
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)



    # 生成Dataloader
    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    
    print("数据集已加载")

    return train_loader, val_loader, test_loader, x_temp, x_freq, N

def get_X_Freq(seq):
    # *** 数据分割
    T, N = seq.shape
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    seq_train, seq_val, seq_test = seq[train_indices], seq[val_indices], seq[test_indices]

    # *** 参数设置
    K = 50  # DCT基数
    J = 3  # 贪婪选择的周期数

    # *** 离散余弦变换 (DCT)
    dct_coefficients = np.zeros((K, N))  # 保存 top K 的 DCT 系数
    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行 DCT
        dct_result = dct(seq_train[:, i], type=2, norm='ortho')

        # 计算 DCT 系数的幅值
        dct_magnitude = np.abs(dct_result)
        top_k_indices = np.argsort(dct_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 计算初始 top-k 周期值
        top_k_periods = [T / k if k != 0 else float('inf') for k in top_k_indices]  # k=0 时周期为无穷大

        # 保存 top K 系数
        dct_coefficients[:K, i] = dct_result[top_k_indices]

        # 贪婪选择最优的 J 个周期
        selected_freqs = []
        selected_amps = []
        selected_phases = []

        for j in range(J):
            best_freq = None
            best_amp = None
            best_phase = None
            min_val_error = float('inf')  # 初始化最小验证误差为无穷大

            for k in top_k_indices:
                if k in selected_freqs:  # 跳过已选择的频率
                    continue

                # 当前频率对应的 DCT 系数
                coeff = dct_result[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k / T  # 归一化频率

                # 重建信号
                t_train = np.arange(train_size)
                reconstructed_signal = np.zeros_like(t_train, dtype=np.float64)

                # 加上已选频率和当前频率对应的信号分量
                for f, a, p in zip(selected_freqs + [freq], selected_amps + [amp], selected_phases + [phase]):
                    reconstructed_signal += a * np.cos(2 * np.pi * f * t_train + p)

                # 加上直流分量 A₀
                A0 = dct_result[0] / np.sqrt(train_size)  # DCT 系数的直流分量
                reconstructed_signal += A0

                # 计算验证集误差
                t_val = np.arange(val_size)
                reconstructed_val = reconstructed_signal[:val_size]
                val_error = np.mean((reconstructed_val - seq_val[:, i]) ** 2)

                # 贪婪选择当前最优的频率
                if val_error < min_val_error:
                    min_val_error = val_error
                    best_freq = freq
                    best_amp = amp
                    best_phase = phase

            # 保存当前最优选择
            selected_freqs.append(best_freq)
            selected_amps.append(best_amp)
            selected_phases.append(best_phase)

        # 存储贪婪选择的频率、幅值和相位
        selected_periods.append((selected_freqs, selected_amps, selected_phases))

        # 计算贪婪选择后的周期值
        # greedy_selected_periods = [T / (freq * T) for freq in selected_freqs]

        greedy_selected_periods = []
        for freq in selected_freqs:
            if freq == 0:
                greedy_selected_periods.append(float('inf'))
            else:
                greedy_selected_periods.append(T / (freq * T))
    
    z_list = []
    # 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的频率、幅值和相位
        selected_freqs, selected_amps, selected_phases = selected_periods[i]

        # 重建训练集信号
        t_train = np.arange(train_size)
        reconstructed_train = np.zeros_like(t_train, dtype=np.float64)

        # 加上直流分量 A₀
        A0 = dct(seq_train[:, i], type=2, norm='ortho')[0] / np.sqrt(train_size)
        reconstructed_train += A0

        for j in range(J):
            freq = selected_freqs[j]
            amp = selected_amps[j]
            phase = selected_phases[j]
            reconstructed_train += amp * np.cos(2 * np.pi * freq * t_train + phase)
        z_list.append(reconstructed_train)
        
    z_t = np.array(z_list)
    z_t = z_t.T
    z_t_mean = z_t.mean(axis=0)
    return z_t_mean

# 指标计算
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

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

# 最大最小归一化
def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val

# 加载数据集
train_loader, val_loader, test_loader, temp, freq, N = load_data(speed_path, od_path, seed, BATCH)
#%% md
# ## 实例化模型
#%%
# 定义的MLP层的神经元数量
n1,n2 = 128, 64

# OD模型
class ODModel(nn.Module):
    def __init__(self, N, temp, freq, n1, n2):
        super(ODModel, self).__init__()

        self.N = N  # 区域数
        self.temp = temp  # 时域差距 [N,]
        self.freq = freq  # 频域差距 [N,]
        self.n1 = n1  # 隐藏层神经元1
        self.n2 = n2  # 隐藏层神经元2

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
        production = x + torch.sum(tensor_delta_x * self.weights.unsqueeze(0), dim=2)


        # 输入到 MLP 网络中
        od_matrix_flat = self.mlp(production)  # [B,N] --> [B, N * N]
        # reshape 最终输出
        od_matrix = od_matrix_flat.view(B, N, N)  # [B, N, N]

        return od_matrix

# 加载模型
model = ODModel(N, temp, freq, n1, n2)
print("模型已加载")
#%% md
# ## 模型训练
#%%
# 模型参数保存路径
ckpt_path = 'ckpt/RelationalEquation/'

# 训练过程
def train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=0.001, ckpt_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("使用的device：", device)

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

        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
                
            torch.save(model.state_dict(), ckpt_path+"best_model.pth")
            if epoch > 0:
                print(f"best model saved at epoch {epoch + 1}, valid loss：{best_val_loss:.4f}")
            
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
# 0.05  0.04  0.03  0.025 0.02  0.01       0.005 0.004 0.003 0.002 0.001
# 39.84 39.0  44    39.13   47   60          55     55    56    56    56
# 开始训练
train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=0.04, ckpt_path=ckpt_path)
#%% md
# ## 模型推理
# ## 可选择测试集推理（type=0）或实时样本推理（type=1）
#%%
# 自定义进行测试集推理(0)还是实时样本推理(1)

# 测试集推理
infer_type = 0

# 实时样本推理
# infer_type = 1
realtime_path = 'data/realtime_speed_25.3.13.npy'  # [N,]
sample = np.load(realtime_path)


# 模型参数保存路径
ckpt_path = 'ckpt/RelationalEquation/'

# 测试集测试
def test_model(model, test_loader, ckpt_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path + "best_model.pth"))

    model.eval()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 计算 RMSE 和 MAE
            rmse, mae = calculate_rmse_mae(outputs * mask, targets)
            rmse_total += rmse
            mae_total += mae

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f}")
    torch.save(model.state_dict(),ckpt_path+f"best_model_feature4_25.1.14数据集版本_{test_loss:.4f}_{rmse_total:.4f}_{mae_total:.4f}.pth")

    # 计算平均的OD矩阵
    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)
    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)

    vmin = min(all_pred_od.min(), all_real_od.min())
    vmax = max(all_pred_od.max(), all_real_od.max())
    

    plt.figure(figsize=(15, 7))
    
    # 绘制真实 OD 热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(all_real_od, cmap="Blues", cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average True OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 绘制预测 OD 热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(all_pred_od, cmap="Blues", cbar=True, vmin=vmin, vmax=vmax)
    plt.title("Average Predicted OD Matrix", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# 实时样本推理
def realtime_infer(model, sample, ckpt_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path + "best_model.pth"))

    model.eval()

    T = sample.shape[0]
    x = np.arange(0, T)

    # 绘制折线图
    plt.figure(figsize=(10, 5))
    plt.plot(x, sample, marker='o')

    # 添加图表标题和轴标签
    plt.title('Speed Data of Network')
    plt.xlabel('RegionID')
    plt.ylabel('Speed')

    # 显示图形
    plt.show()
    
    scaler = MinMaxScaler()
    sample = scaler.fit_transform(sample.reshape(-1, 1))  # [N,1]
    sample = np.squeeze(sample, axis=-1).astype(np.float32)  # [N,]
    
    sample = torch.from_numpy(np.expand_dims(sample, axis=0))  # [1,N]
    

    with torch.no_grad():
        input = sample.to(device)
        
        # 设置对角线掩码
        mask = torch.ones(N, N).to(device)
        for i in range(N):
            mask[i, i] = 0
        output = model(input)
        prediction = (output.squeeze(dim=0) * mask).cpu().numpy()

    vmin = prediction.min()
    vmax = prediction.max()
    

    plt.figure(figsize=(8, 7))
    

    # 绘制预测OD热力图
    plt.subplot(1, 1, 1)
    sns.heatmap(prediction, cmap="Blues", cbar=True, vmin=prediction.min(), vmax=prediction.max())
    plt.title("Real-time OD Matrix Prediction", fontsize=14)
    plt.xlabel("Destination Zones")
    plt.ylabel("Origin Zones")

    # 调整布局并显示
    # plt.tight_layout()
    # plt.show()

if infer_type == 0:
    test_model(model, test_loader, ckpt_path=ckpt_path)
else:
    realtime_infer(model, sample, ckpt_path=ckpt_path)
#%%

#%%
