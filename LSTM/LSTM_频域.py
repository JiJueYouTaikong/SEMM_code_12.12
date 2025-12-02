# import numpy as np
#
# speed_data = np.load('data/WH_Speed_1km_110region.npy')
# speed_data = speed_data[:, :, 0]  # 形状为[168, 110]
# od_data = np.load('data/WH_OD_1km_110region_1.npy')        # 形状为[168, 110, 110]
# freq = np.load('data/WH_Freq_1km.npy') # [168,110]
#
# # 合并 speed_data 和 freq 为形状 [168, 110, 2]
# combined_data = np.stack((speed_data, freq), axis=-1)
# np.save('data/WH_Speed_1km_110region_Freq.npy', combined_data)

import logging
from scipy import stats
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)  # [batch_size, T, 110*2]

        # print(x.shape)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# 2. 数据加载与预处理
# 加载数据
speed_data = np.load('data/WH_Speed_1km_110region_Freq.npy')  # 形状为[168, 110]
# speed_data = speed_data[:, :, 0]
od_data = np.load('data/WH_OD_1km_110region_1.npy')        # 形状为[168, 110, 110]



# 对 [168, 110, 110] 的数据在第一个维度（时间步）上求均值
average_data = np.mean(od_data, axis=0)

# # YlGnBu twilight Blues
# style =  'YlGnBu'
style =  'Blues'
# 2. 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(average_data, cmap=style, cbar=True)
plt.title('Average Heatmap of OD Data Over Time Steps')
plt.xlabel('Region Index')
plt.ylabel('Region Index')
plt.show()



# 归一化速度数据
# scaler = MinMaxScaler()
# speed_data_normalized = scaler.fit_transform(speed_data.reshape(-1, speed_data.shape[1])).reshape(speed_data.shape)

# 归一化速度数据
scaler = MinMaxScaler()
# 对两个特征分别进行归一化处理
speed_data_normalized = np.zeros_like(speed_data)
for i in range(speed_data.shape[2]):  # 遍历两个特征
    speed_data_normalized[:, :, i] = scaler.fit_transform(speed_data[:, :, i])


# 创建输入和目标数据
T = 3
def create_sequences(speed_data, od_data, T):
    inputs, targets = [], []
    for i in range(len(speed_data) - T):
        inputs.append(speed_data[i:i+T])
        targets.append(od_data[i:i+T])
    return np.array(inputs), np.array(targets)

X, y = create_sequences(speed_data_normalized, od_data, T)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42)

# 然后再从临时集划分出验证集和测试集（按照 2:2 划分）
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
print(X_train.shape, X_val.shape, X_test.shape)

# 转换为Tensor
train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
print(len(train_data), len(val_data), len(test_data))

# 数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 3. 初始化模型、损失函数和优化器
model = LSTMModel(input_size=110*2, hidden_size=128, output_size=110*110, num_layers=2)

learning_rate = 0.05
epochs = 1000
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
patience = 10 # 允许验证损失没有改善的周期数
early_stop_counter = 0  # 计数器

# 4. 设置日志记录
logging.basicConfig(filename='log/training_频域.log', level=logging.INFO,
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
    print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')
    logging.info(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss / len(val_loader):.2f}')

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'ckpt/best_model_频域.pth')
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
plt.title('Training and Validation Loss')
plt.show()

# 7. 测试过程
model.load_state_dict(torch.load('ckpt/best_model_频域.pth'))
logging.info('load the best model, start testing')

# 计算 RMSE 和 MAE，并保存到日志
total_rmse = 0.0
total_mae = 0.0
real_od_sum = torch.zeros(110, 110)  # 初始化真实OD的累计矩阵
predicted_od_sum = torch.zeros(110, 110)  # 初始化预测OD的累计矩阵
total_samples = 0  # 累计样本数

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.view(targets.size(0), targets.size(1), -1).to(device)
        outputs = model(inputs)  # 模型推理



        # 计算 RMSE 和 MAE
        rmse = torch.sqrt(((outputs - targets) ** 2).mean()).item()
        mae = (torch.abs(outputs - targets)).mean().item()
        total_rmse += rmse
        total_mae += mae

        outputs = outputs.view(-1, 110, 110)  # 恢复为 [batch, T, 110, 110] 的矩阵
        targets = targets.view(-1, 110, 110)
        # 累计真实和预测的 OD 矩阵
        real_od_sum += targets.sum(dim=0)  # 按 batch 累加
        predicted_od_sum += outputs.sum(dim=0)
        total_samples += inputs.size(0)

# 计算最终的 RMSE 和 MAE
rmse = total_rmse / len(test_loader)
mae = total_mae / len(test_loader)
print(f'Test RMSE: {rmse}, Test MAE: {mae}')
logging.info(f"learning rate: {learning_rate}")
logging.info(f'Test RMSE: {rmse}, Test MAE: {mae}')

# 计算平均真实 OD 和预测 OD
real_od_avg = real_od_sum / (total_samples * T)
real_od_avg = real_od_avg.cpu().numpy()

predicted_od_avg = predicted_od_sum / (total_samples * T)
predicted_od_avg = predicted_od_avg.cpu().numpy()


#绘制线性拟合
# 计算拟合线
slope, intercept, r_value, p_value, std_err = stats.linregress(real_od_avg.flatten(), predicted_od_avg.flatten())

# 绘制散点图和拟合线
plt.figure(figsize=(8, 6))

# 绘制真实样本点（红色）
plt.scatter(real_od_avg.flatten(), predicted_od_avg.flatten(), c='red', label='Real OD Average')

# 绘制预测样本点（蓝色）
plt.scatter(predicted_od_avg.flatten(), real_od_avg.flatten(), c='blue', label='Predicted OD Average')

# 绘制拟合线
plt.plot(real_od_avg.flatten(), slope * real_od_avg.flatten() + intercept, color='green', label=f'Fit line: y = {slope:.3f}x + {intercept:.3f}')

# 绘制图形
plt.xlabel('Real OD Average')
plt.ylabel('Predicted OD Average')
plt.title('Real vs Predicted OD Average with Fit Line')
plt.legend()
plt.show()

# 输出拟合线方程
print(f'Fit Line Equation: y = {slope:.3f}x + {intercept:.3f}')





print(real_od_avg[40:44,40:48].astype(int))
print("--------------------------------------------------")
print(predicted_od_avg[40:44,40:48].astype(int))

# 前两个图：展示 0-55 序号的 OD 矩阵的真实和预测值
start1, end1 = 0, 56
# start1, end1 = 0, 40

sub_real_1 = real_od_avg[start1:end1, start1:end1]
sub_pred_1 = predicted_od_avg[start1:end1, start1:end1]

# 后两个图：展示 56-110 序号的 OD 矩阵的真实和预测值
start2, end2 = 56, 111
# start2, end2 = 40, 80

sub_real_2 = real_od_avg[start2:end2, start2:end2]
sub_pred_2 = predicted_od_avg[start2:end2, start2:end2]

# 统一量纲范围
vmin1 = min(sub_real_1.min(), sub_pred_1.min())
vmax1 = max(sub_real_1.max(), sub_pred_1.max())

vmin2 = min(sub_real_2.min(), sub_pred_2.min())
vmax2 = max(sub_real_2.max(), sub_pred_2.max())


# 绘制热力图
plt.figure(figsize=(18, 8))



# 前两个图展示 0-55 序号的真实和预测 OD 矩阵
plt.subplot(1, 2, 1)
sns.heatmap(sub_real_1, cmap=style, cbar=True, vmin=vmin1, vmax=vmax1)
plt.title('Real OD Heatmap (0-55)')

plt.subplot(1, 2, 2)
sns.heatmap(sub_pred_1, cmap=style, cbar=True, vmin=vmin1, vmax=vmax1)
plt.title('Predicted OD Heatmap (0-55)')

plt.tight_layout()
plt.show()

# 绘制热力图
plt.figure(figsize=(18, 8))
# 后两个图展示 56-110 序号的真实和预测 OD 矩阵
plt.subplot(1, 2, 1)
sns.heatmap(sub_real_2, cmap=style, cbar=True, vmin=vmin2, vmax=vmax2)
plt.title('Real OD Heatmap (56-110)')

plt.subplot(1, 2, 2)
sns.heatmap(sub_pred_2, cmap=style, cbar=True, vmin=vmin2, vmax=vmax2)
plt.title('Predicted OD Heatmap (56-110)')
plt.tight_layout()
plt.show()