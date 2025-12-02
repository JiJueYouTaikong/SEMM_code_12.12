import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子



# 损失函数：忽略观测值为 0 的位置
def traffic_volume_loss(estimated_volume, observed_volume):
    assert estimated_volume.shape == observed_volume.shape
    mask = observed_volume != 0
    valid_estimated = torch.masked_select(estimated_volume, mask)
    valid_observed = torch.masked_select(observed_volume, mask)
    diff_ratio = (valid_estimated - valid_observed) / valid_observed
    squared_diff = diff_ratio ** 2
    loss = torch.sum(squared_diff)
    return loss

# RMSE, MAE, MAPE
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


def compute_weighted_sum_from_x(x, model, temp, freq):
    """
    根据输入的x向量，通过模型的weights计算得到weighted_sum向量

    参数:
    - x: 形状为[B, N]的输入向量
    - model: 加载了最佳参数的ODModel模型
    - temp: 时间特征向量，形状为[N,]
    - freq: 频率特征向量，形状为[N,]

    返回:
    - weighted_sum: 加权后的向量，形状为[B, N]
    """
    device = model.weights.device
    x = x.to(device)
    B, N = x.shape

    # 保证 temp 和 freq 为 numpy 数组
    temp = np.asarray(temp)
    freq = np.asarray(freq)


    # 构造 [B, N, 3] 的辅助特征张量
    x_temp = np.tile(temp, (B, 1))
    x_freq = np.tile(freq, (B, 1))
    x_random = np.clip(np.random.normal(0.05, 0.01, size=(B,N)), 0, 0.1)

    tensor_delta_x = np.stack([x_temp, x_freq, x_random], axis=-1)
    tensor_delta_x = torch.tensor(tensor_delta_x, dtype=torch.float32, device=device)

    # 获取模型的权重 [N, 3] -> [1, N, 3]
    weights = model.weights.unsqueeze(0)  # [1, N, 3]

    print("Model weights:", model.weights.detach().cpu().numpy())

    # 计算 delta 项并加到 x 上
    delta_term = torch.sum(tensor_delta_x * weights, dim=2)  # [B, N]
    weighted_sum = x + delta_term

    return weighted_sum


def infer_x_from_weighted_sum(weighted_sum, model, temp, freq,random=None):
    """
    从weighted_sum向量反推得到x向量

    参数:
    - weighted_sum: 形状为[B=1, N]的weighted_sum向量
    - model: 加载了最佳参数的ODModel模型
    - temp: 时间特征向量
    - freq: 频率特征向量

    返回:
    - x: 反推得到的x向量，形状为[B=1, N]
    """

    weighted_sum = torch.unsqueeze(weighted_sum,dim=0)
    device = model.weights.device

    B, N = weighted_sum.shape

    # 确保weighted_sum在与模型相同的设备上
    weighted_sum = weighted_sum.to(device)

    # 构造tensor_delta_x
    x_temp_repeated = np.tile(temp, (B, 1))
    x_freq_repeated = np.tile(freq, (B, 1))
    random = np.tile(random, (B, 1))

    tensor_delta_x = np.stack([x_temp_repeated, x_freq_repeated, random], axis=-1)
    tensor_delta_x = torch.tensor(tensor_delta_x, dtype=torch.float32, device=device)

    # 获取模型权重
    weights = model.weights.unsqueeze(0)  # [1, N, 3]

    # 计算sum(tensor_delta_x * weights, dim=2)
    delta_term = torch.sum(tensor_delta_x * weights, dim=2)  # [B, N]


    # 反推x
    x = weighted_sum - delta_term

    return x


def get_n_p_pi_from_weighted_sum(weighted_sum, model):
    """
    从weighted_sum向量得到n, p, pi矩阵

    参数:
    - weighted_sum: 形状为[B=1, N]的weighted_sum向量
    - model: 加载了最佳参数的ODModel模型

    返回:
    - n, p, pi: 形状均为[B=1, N, N]的张量
    """
    device = model.weights.device
    B, N = weighted_sum.shape

    # 将weighted_sum传入MLP网络
    n_flat = model.mlp_n(weighted_sum)
    p_flat = model.mlp_p(weighted_sum)
    pi_flat = model.mlp_pi(weighted_sum)

    # 调整形状并应用激活函数
    n = F.softplus(n_flat.view(B, N, N))
    p = torch.sigmoid(p_flat.view(B, N, N))
    pi = torch.sigmoid(pi_flat.view(B, N, N))

    return n, p, pi

class ODModel(nn.Module):
    def __init__(self, N, temp, freq, num_layers=8, hidden_units=256, dropout_rate=0.0):
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


# 模型：直接输出 OD 作为预测 flow
class SSMModel(nn.Module):
    def __init__(self, N, init_p_vector):
        super(SSMModel, self).__init__()
        self.pe = nn.Parameter(torch.from_numpy(init_p_vector).float())  # [N,]

    def forward(self):
        '''
        weighted_sum --> [1,N]
        :return: speed --> [1,N]
        '''

        x = infer_x_from_weighted_sum(self.pe, model, temp, freq,x_random_cache)
        return x


# ------------------ 初始化RED模型 ------------------

set_seed(42)

# x_random_cache = np.clip(np.random.normal(0.05, 0.01, size=(110,)), 0, 0.1)
x_random_cache = np.load("../data/x_random.npy")
# print(f"x_random_shape:{x_random_cache.shape}")
x_random_cache = x_random_cache[0]
temp = np.load("../data/temp.npy")
freq = np.load("../data/freq.npy")

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = ODModel(N=110, temp=temp, freq=freq)

model.load_state_dict(torch.load("ckpt/best_model_WT_is_mcm_True.pth"))

model = model.to(device)




# ------------------ 数据加载 ------------------
N = 110
num_samples = 35

# 加载原始数据
flow_all = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy').astype(np.float32)  # [T, N]
od_all = np.load('../data/OD_完整批处理_3.17_Final.npy').astype(np.float32)      # [T, N, N]


flow_train_raw = flow_all[:1400]                 # [1400, 110]

flow_test_raw = flow_all[-num_samples:]         # [35, 110]
od_test_raw = od_all[-num_samples:]             # [35, 110, 110]

# 2. 构建 scaler 并拟合训练集
scaler = MinMaxScaler()

# 3. 归一化数据
flow_train_norm = scaler.fit_transform(flow_train_raw.reshape(-1, 1)).reshape(flow_train_raw.shape)  # [1400, 110]
flow_test_norm = scaler.transform(flow_test_raw.reshape(-1, 1)).reshape(flow_test_raw.shape)    # [35, 110]
flow = flow_test_norm  # [35,110]
print(flow[0, :10])
od = od_test_raw  # [35,110,110]

input_x = flow_train_norm  # [1400,110]

init_type = 3
# ------------------ 获取P向量的初始值 方式1 方式2------------------
init_p_vector = np.random.rand(N).astype(np.float32)               # 初始化1 [110]，生成0-1之间的随机数
# init_p_vector = np.zeros(N, dtype=np.float32)                    # 初始化2 [110]


# ------------------ 获取P向量的初始值 方式2 ------------------
# input_x = torch.from_numpy(input_x).to(device)
# init_p_vectors = compute_weighted_sum_from_x(input_x, model, temp, freq)
# # init_p_vector = torch.squeeze(init_p_vectors,dim=0)
# init_p_vector = torch.mean(init_p_vectors, dim=0)
# init_p_vector = init_p_vector.cpu().detach().numpy()             # 初始化3 [110]

# ------------------ 参数设置 ------------------
num_epochs = 100000
tol = 1e-5
learning_rate = 0.001

rmse_total = 0
mae_total = 0
mape_total = 0
cpc_total = 0
jsd_total = 0
loss_total = 0

start_time = time.time()
all_predicted_ods = []

# ------------------ 训练流程 ------------------
for i in range(num_samples):
    print(f"Sample {i+1}/{num_samples}", flush=True)
    flow_tensor = torch.tensor(flow[i], dtype=torch.float32,device=device)

    ssmmodel = SSMModel(N, init_p_vector)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ssmmodel.parameters(), lr=learning_rate)

    print(f"真实OD总量:{od[i].sum():.2f}")
    print(f"初始P总量:{ssmmodel.pe.sum():.2f}")

    od_tensor = torch.tensor(od[i], dtype=torch.float32, device=device)
    p_true = np.sum(od[i], axis=-1)
    p_true_tensor = torch.tensor(p_true, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        flow_pred = ssmmodel() # [N,]
        flow_pred = flow_pred.squeeze(0)

        # loss = criterion(flow_pred, flow_tensor)
        loss = traffic_volume_loss(flow_pred, flow_tensor)




        if loss <= tol:
            print(f"[Early Stop] Loss reached tol: {loss:.4f}")
            break

        loss.backward()
        optimizer.step()
        if epoch % 50 ==0 :
            print(f"[Epoch {epoch}] Loss: {loss.item():.8f} 真实OD总量:{od[i].sum():.2f} PE总量: {ssmmodel.pe.sum():.2f}")



    pred_pe = ssmmodel.pe.detach()
    pred_pe = pred_pe.unsqueeze(0)  # [1,N]
    pred_pe = pred_pe.to(device)

    # 计算n, p, pi
    n, p, pi = get_n_p_pi_from_weighted_sum(pred_pe, model)

    # 计算预测值
    pred_od = (1 - pi.detach().cpu().numpy()) * (
                n.detach().cpu().numpy() / p.detach().cpu().numpy() - n.detach().cpu().numpy())


    pred_od = torch.tensor(pred_od, dtype=torch.float32).to(device)

    pred_od = pred_od.squeeze(0)

    all_predicted_ods.append(pred_od.cpu().numpy())

    # loss_total += loss.item()

    rmse, mae, mape, cpc, jsd = calculate_rmse_mae(pred_od, od_tensor)

    print(f"[Sample {i}] RMSE: {rmse:.4f} MAE: {mae:.4f} MAPE: {mape:.4f}")
    rmse_total += rmse
    mae_total += mae
    mape_total += mape
    cpc_total += cpc
    jsd_total += jsd

# ------------------ 评估和保存 ------------------
rmse_test = rmse_total / num_samples
mae_test = mae_total / num_samples
mape_test = mape_total / num_samples
cpc_test = cpc_total / num_samples
jsd_test = jsd_total / num_samples
# loss_test = loss_total / num_samples

# np.save('../可视化/测试集TNN/Pred_SSM_Grid_6.15.npy', np.array(all_predicted_ods))
# print(f"[INFO] 所有预测的OD矩阵已保存")

print(f"[INFO] RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.4f}")
print(f"总耗时: {(time.time() - start_time) / 60:.2f} min")

with open("log/SSM_GRID.log", 'a') as f:
    f.write(
        f"Samples: {num_samples}, init_type:{init_type} Epochs: {num_epochs}, Lr: {learning_rate}, Tol: {tol},  RMSE: {rmse_test:.4f} MAE: {mae_test:.4f} MAPE: {mape_test:.4f} CPC: {cpc_test:.4f} JSD: {jsd_test:.4f}, Time: {(time.time() - start_time)/60:.2f} min\n")
