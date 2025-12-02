import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import get_data

# 定义模型（与之前相同）
class TODGenerator(nn.Module):
    def __init__(self, noise_dim, od_dim):
        super(TODGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128, od_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.sigmoid(self.fc1(z))
        od = self.sigmoid(self.fc2(h))
        return od

class TODVolumeMapping(nn.Module):
    def __init__(self, od_dim, link_dim):
        super(TODVolumeMapping, self).__init__()
        self.fc_route = nn.Linear(od_dim, link_dim)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.fc_att = nn.Linear(64, link_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, od):
        p = self.fc_route(od)
        p = p.unsqueeze(1)
        h1 = self.conv1(p)
        h2 = self.conv2(h1)
        e = h2.mean(dim=2)
        alpha = self.softmax(self.fc_att(e))
        alpha = alpha.unsqueeze(2)
        flow = torch.sum(alpha * p, dim=1)
        return flow

class VolumeSpeedMapping(nn.Module):
    def __init__(self, link_dim, hidden_size):
        super(VolumeSpeedMapping, self).__init__()
        self.lstm1 = nn.LSTM(link_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, link_dim)

    def forward(self, flow):
        flow = flow.unsqueeze(1)
        h1, _ = self.lstm1(flow)
        h2, _ = self.lstm2(h1)
        speed = self.fc(h2.squeeze(1))
        return speed

class OVSModel(nn.Module):
    def __init__(self, noise_dim, od_dim, link_dim, hidden_size):
        super(OVSModel, self).__init__()
        self.tod_generator = TODGenerator(noise_dim, od_dim)
        self.tod_volume_mapping = TODVolumeMapping(od_dim, link_dim)
        self.volume_speed_mapping = VolumeSpeedMapping(link_dim, hidden_size)

    def forward(self, z):
        od = self.tod_generator(z)
        flow = self.tod_volume_mapping(od)
        speed = self.volume_speed_mapping(flow)
        return od, flow, speed



def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))
    return rmse.item(), mae.item()
    

def train_model(model, train_loader, val_loader, test_loader,epochs, patience, learning_rate):
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_flow_speed = optim.Adam(model.volume_speed_mapping.parameters(), lr=learning_rate)
    optimizer_od_flow = optim.Adam(model.tod_volume_mapping.parameters(), lr=learning_rate)
    optimizer_od_gen = optim.Adam(model.tod_generator.parameters(), lr=learning_rate)

    # 训练阶段
    print("Training Volume-Speed Mapping...")
    for epoch in range(epochs):
        for _, flow, speed in train_loader:
            # 训练 Volume-Speed Mapping
            optimizer_flow_speed.zero_grad()
            speed_pred = model.volume_speed_mapping(flow)
            loss = criterion(speed_pred, speed)
            loss.backward()
            optimizer_flow_speed.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training TOD-Volume Mapping...")
    for epoch in range(epochs):
        for od, _, speed in train_loader:
            # 训练 TOD-Volume Mapping
            optimizer_od_flow.zero_grad()
            flow_pred = model.tod_volume_mapping(od)
            speed_pred = model.volume_speed_mapping(flow_pred)
            loss = criterion(speed_pred, speed)
            loss.backward()
            optimizer_od_flow.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


    # 测试阶段
    print("Testing TOD Generation...")

    test_loss = 0
    rmse_total = 0
    mae_total = 0

    all_real_od = []
    all_pred_od = []

    # 固定 TOD-Volume Mapping 和 Volume-Speed Mapping 参数
    for param in model.tod_volume_mapping.parameters():
        param.requires_grad = False
    for param in model.volume_speed_mapping.parameters():
        param.requires_grad = False

    # 微调 TOD Generation
    for epoch in range(epochs):
        for z_test, _, _, speed_test in test_loader:
            optimizer_od_gen.zero_grad()
            od_pred = model.tod_generator(z_test)
            flow_pred = model.tod_volume_mapping(od_pred)
            speed_pred = model.volume_speed_mapping(flow_pred)
            loss = criterion(speed_pred, speed_test)
            loss.backward()
            optimizer_od_gen.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 最终重建的 TOD
    for z_test, od_test, _, speed_test in test_loader:
        od_pred = model.tod_generator(z_test)

        loss = criterion(od_pred, od_test)
        test_loss += loss.item()

        # 计算 RMSE 和 MAE
        rmse, mae = calculate_rmse_mae(od_pred, od_test)
        print(f"batch的RMSE和MAE:{rmse:.4f},{mae:.4f}")
        rmse_total += rmse
        mae_total += mae

        all_real_od.append(od_test.cpu().numpy())
        all_pred_od.append(od_pred.cpu().numpy())


    print(f"总RMSE和MAE：{rmse_total:.4f},{mae_total:.4f}")
    print(f"测试集大小：{len(test_loader)}")
    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f}")

    # 计算平均的OD矩阵
    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)
    print(f"所有时间步的OD预测：{all_real_od_t.shape}")
    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)
    print(f"时间步平均后的OD预测：{all_real_od.shape}")

    # g_reconstructed = model.tod_generator(z_test)
    # print("Reconstructed TOD shape:", g_reconstructed.shape)

# 主程序
def main():
    train_loader, val_loader, test_loader,T,N,L= get_data.load_data()

    # 超参数
    noise_dim = N
    od_dim = N
    link_dim = L
    hidden_size = 64
    epochs = 200
    learning_rate = 0.001

    # 初始化模型
    model = OVSModel(noise_dim, od_dim, link_dim, hidden_size)

    lr = 0.001 # best
    train_model(model, train_loader, val_loader,test_loader, epochs=200, patience=20, learning_rate=lr)



    # # 测试模型
    # test_model(model, test_loader,z,epochs=200, patience=20, learning_rate=lr)


if __name__ == "__main__":
    main()