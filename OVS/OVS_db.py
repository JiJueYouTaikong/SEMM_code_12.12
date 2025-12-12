import torch
import torch.nn as nn


class TODGeneration(nn.Module):
    def __init__(self, num_od_pairs, num_time_intervals):
        super(TODGeneration, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_od_pairs, 16),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16, num_od_pairs * num_time_intervals),
            nn.Sigmoid()
        )
        self.num_time_intervals = num_time_intervals

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x.view(-1, num_od_pairs, self.num_time_intervals)


class TODVolumeMapping(nn.Module):
    def __init__(self, num_od_pairs, num_time_intervals):
        super(TODVolumeMapping, self).__init__()
        self.od_route = nn.Sequential(
            nn.Linear(num_od_pairs * num_time_intervals, 16),
            nn.Sigmoid()
        )
        self.route_e = nn.Sequential(
            nn.Conv1d(num_od_pairs, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.e_alpha = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_od_route = self.od_route(x.view(-1, num_od_pairs * num_time_intervals))
        x_route_e = self.route_e(x_od_route.view(-1, num_od_pairs, num_time_intervals))
        x_e_alpha = self.e_alpha(x_route_e.mean(dim=2))
        return x_e_alpha.unsqueeze(2) * x_od_route.view(-1, num_od_pairs, num_time_intervals)


class VolumeSpeedMapping(nn.Module):
    def __init__(self, num_links, num_time_intervals):
        super(VolumeSpeedMapping, self).__init__()
        self.lstm1 = nn.LSTM(num_links, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, num_links)
        )
        self.num_time_intervals = num_time_intervals

    def forward(self, x):
        # 调整输入形状以适应LSTM的输入要求，假设输入x形状为[batch_size, num_links]
        x = x.unsqueeze(1).repeat(1, self.num_time_intervals, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x


class OVS(nn.Module):
    def __init__(self, num_od_pairs, num_links, num_time_intervals):
        super(OVS, self).__init__()
        self.tod_generation = TODGeneration(num_od_pairs, num_time_intervals)
        self.tod_volume_mapping = TODVolumeMapping(num_od_pairs, num_time_intervals)
        self.volume_speed_mapping = VolumeSpeedMapping(num_links, num_time_intervals)

    def forward(self, x):
        tod = self.tod_generation(x)
        volume = self.tod_volume_mapping(tod)
        speed = self.volume_speed_mapping(volume)
        return speed


# 参数设置
num_od_pairs = 10
num_links = 20
num_time_intervals = 5
batch_size = 64

model = OVS(num_od_pairs, num_links, num_time_intervals)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 生成虚拟输入数据，形状为[Batchsize, num_od_pairs]
input_seed = torch.randn(batch_size, num_od_pairs)
# 假设的真实速度数据，形状为[batch_size, num_links]
groundtruth_speed = torch.randn(batch_size, num_links)

for epoch in range(10000):
    model.train()
    output_speed = model(input_seed)
    loss = criterion(output_speed, groundtruth_speed)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
