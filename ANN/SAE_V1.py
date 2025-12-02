import os

import numpy as np
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# 稀疏自编码器
class SparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rho=0.05, beta=4.0, lam=0.04):
        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.rho = rho  # 稀疏目标
        self.beta = beta
        self.lam = lam

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x))  # 用于KL散度
        out = self.decoder(z)  # 线性激活
        return out, z

    def loss(self, x, out, hidden):
        mse = F.mse_loss(out, x, reduction='mean')

        # 稀疏约束
        rho_hat = torch.mean(hidden, dim=0)
        kl = self.rho * torch.log(self.rho / (rho_hat + 1e-8)) + \
             (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat + 1e-8))
        kl_loss = torch.sum(kl)

        # L2 正则
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)

        return mse + self.beta * kl_loss + self.lam * l2_loss

# 论文模型：SAE + FCL
class SAEOFCLModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[10, 10, 10],hid_dim=None):
        super(SAEOFCLModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.autoencoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.N = input_dim

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            sae = SparseAutoEncoder(dims[i], dims[i+1])
            self.autoencoders.append(sae)
            self.encoders.append(sae.encoder)

        # 最后的FCL输出层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, output_dim)
        )

    def forward(self, x):
        for encoder in self.encoders:
            x = torch.sigmoid(encoder(x))
            out = self.predictor(x)
            out = out.view(-1,self.N, self.N)
        return out

# SAE预训练（无监督，逐层）
def pretrain_saes(model, data, epochs=2000, lr=0.01, patience=20, device='cpu'):
    x = data.to(device)
    print("用于预训练的x", x.shape)

    for i, ae in enumerate(model.autoencoders):
        ae.to(device)
        optimizer = optim.Adam(ae.parameters(), lr=lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            ae.train()
            out, hidden = ae(x)
            loss = ae.loss(x, out, hidden)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'[SAE-{i+1}] Epoch {epoch+1}, Loss: {loss.item():.4f}')

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                torch.save(ae.state_dict(), f'ckpt/sae_layer_{i+1}_best.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'[SAE-{i+1}] Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}')
                break

        # 训练完成后加载最优权重
        ae.load_state_dict(torch.load(f'ckpt/sae_layer_{i+1}_best.pth'))
        # 用编码输出作为下一层输入
        x = torch.sigmoid(ae.encoder(x)).detach()


# 训练过程
def train_model(model, train_loader, val_loader, epochs=100, patience=10,learning_rate=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

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

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/best_sae_model.pth")
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
def load_data(is_mcm):
    log_filename = f"log/SAE_MCM_{is_mcm}.log"
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


    # 打印结果形状
    print("特征处理后的训练集 shape:", speed_train.shape, "OD形状", od_train.shape)
    print("特征处理后的验证集 shape:", speed_val.shape, "OD形状", od_val.shape)
    print("特征处理后的测试集 shape:", speed_test.shape, "OD形状", od_test.shape)

    # 归一化
    scaler = MinMaxScaler()

    x_train_scale = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    x_val_scale = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    x_test_scale = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)

    train_data = x_train_scale
    val_data = x_val_scale
    test_data = x_test_scale

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader,log_filename

import time 
# 测试
def test_model(model, test_loader,lr=0,log_filename=None,ae_dim=None, hid_dim=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_sae_model.pth"))

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
    infer_time = end_time - start_time

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)
    cpc_total /= len(test_loader)
    jsd_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f} CPC: {cpc_total:.4f} JSD: {jsd_total:.4f}")

    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)

    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Lr = {lr},AE_dim: {ae_dim} HID_dim: {hid_dim} Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f} CPC: {cpc_total:.4f} JSD: {jsd_total:.4f} Infer time:{infer_time}s\n")

def main():
    # # 定义学习率列表
    lr_list = [0.1, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005,
               0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0001]

    lr_list = [0.01, 0.005,
               0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001]

    ae_dim_list = [10, 32, 64, 128, 256, 512]
    hid_dim_list = [32, 64, 128, 256, 512, 1024]

    ae_dim_list = [10, 32, 64, 128]
    hid_dim_list = [32, 64, 128, 256]


    lr_list = [0.005]  # 是否当前最佳 YES 6.20  Lr = 0.005,AE_dim: 10 HID_dim: 32
    ae_dim_list = [10]
    hid_dim_list = [32]

    lr_list = [0.004]  # MCM 是否当前最佳 YES 6.20  Lr = 0.004,AE_dim: 128 HID_dim: 256
    ae_dim_list = [128]
    hid_dim_list = [256]

    # 遍历学习率列表
    for ae_dim in ae_dim_list:
        for hid_dim in hid_dim_list:
            for lr in lr_list:
                print(f"当前组合: ae_dim={ae_dim}, hid_dim={hid_dim}, lr={lr}")

                train_loader, val_loader, test_loader,log_filename = load_data(is_mcm=True)

                N=110
                model = SAEOFCLModel(input_dim=N, output_dim=N*N, hidden_dims=[ae_dim, ae_dim, ae_dim],hid_dim = hid_dim)

                # 从 train_loader 中提取所有输入用于预训练
                x_all = torch.cat([xb for xb, _ in train_loader], dim=0)

                pretrain_saes(model, x_all, epochs=10000, lr=lr, patience=20, device='cuda' if torch.cuda.is_available() else 'cpu')

                train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=lr)

                test_model(model, test_loader, lr=lr,log_filename=log_filename,ae_dim=ae_dim, hid_dim=hid_dim)

                # 删除预训练权重文件
                for i in range(len(model.autoencoders)):
                    path = f'ckpt/sae_layer_{i + 1}_best.pth'
                    if os.path.exists(path):
                        os.remove(path)

                # 删除微调保存的最优模型权重
                finetune_path = "ckpt/best_sae_model.pth"
                if os.path.exists(finetune_path):
                    os.remove(finetune_path)

if __name__ == "__main__":
    main()