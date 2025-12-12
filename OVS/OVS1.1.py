import torch
import torch.nn as nn
import torch.optim as optim


# 定义TOD生成模块
class TODGenerator(nn.Module):
    def __init__(self, noise_dim, od_dim):
        super(TODGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128, od_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.sigmoid(self.fc1(z))
        g = self.sigmoid(self.fc2(h))
        return g


# 定义TOD-Volume映射模块
class TODVolumeMapping(nn.Module):
    def __init__(self, od_dim, link_dim):
        super(TODVolumeMapping, self).__init__()
        self.fc_route = nn.Linear(od_dim, link_dim)  # 输出维度调整为 link_dim
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.fc_att = nn.Linear(64, link_dim)  # 输出维度调整为 link_dim
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g):
        p = self.fc_route(g)  # (batch_size, link_dim)
        p = p.unsqueeze(1)  # (batch_size, 1, link_dim)

        # 卷积层处理
        h1 = self.conv1(p)  # (batch_size, 32, link_dim - 2)
        h2 = self.conv2(h1)  # (batch_size, 64, link_dim - 4)

        # 全局平均池化
        e = h2.mean(dim=2)  # (batch_size, 64)

        # 计算注意力权重
        alpha = self.softmax(self.fc_att(e))  # (batch_size, link_dim)

        # 调整 alpha 的维度以匹配 p 的维度
        alpha = alpha.unsqueeze(2)  # (batch_size, link_dim, 1)

        print(f"alpha: {alpha.shape}")
        print(f"p:{p.shape}")

        # 计算道路链接的流量
        q = torch.sum(alpha * p, dim=1)  # (batch_size, link_dim)
        print(f"q:{q.shape}")


        return q


# 定义Volume-Speed映射模块
class VolumeSpeedMapping(nn.Module):
    def __init__(self, link_dim, hidden_size):
        super(VolumeSpeedMapping, self).__init__()
        self.lstm1 = nn.LSTM(link_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, link_dim)

    def forward(self, q):
        # 将 q 的 shape 从 (batch_size, link_dim) 调整为 (batch_size, 1, link_dim)
        q = q.unsqueeze(1)  # (batch_size, 1, link_dim)

        # LSTM 处理
        h1, _ = self.lstm1(q)  # (batch_size, 1, hidden_size)
        h2, _ = self.lstm2(h1)  # (batch_size, 1, hidden_size)

        # 全连接层处理
        v = self.fc(h2.squeeze(1))  # (batch_size, link_dim)
        return v


# 定义完整的OVS模型
class OVSModel(nn.Module):
    def __init__(self, noise_dim, od_dim, link_dim, hidden_size):
        super(OVSModel, self).__init__()
        self.tod_generator = TODGenerator(noise_dim, od_dim)
        self.tod_volume_mapping = TODVolumeMapping(od_dim, link_dim)
        self.volume_speed_mapping = VolumeSpeedMapping(link_dim, hidden_size)

    def forward(self, z):
        g = self.tod_generator(z)
        q = self.tod_volume_mapping(g)
        v = self.volume_speed_mapping(q)
        return g, q, v


# 虚拟数据生成
batch_size = 32
noise_dim = 100
od_dim = 50
link_dim = 20
hidden_size = 64

z = torch.randn(batch_size, noise_dim)  # 随机噪声输入

# 初始化模型
model = OVSModel(noise_dim, od_dim, link_dim, hidden_size)

# 前向传播
g, q, v = model(z)

# 打印输出shape
print("TOD shape:", g.shape)
print("Volume shape:", q.shape)
print("Speed shape:", v.shape)