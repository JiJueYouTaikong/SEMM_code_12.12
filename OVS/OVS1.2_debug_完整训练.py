import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import get_data
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义模型
class TODGenerator(nn.Module):
    def __init__(self, noise_dim, od_dim):
        super(TODGenerator, self).__init__()
        print("noise_dim", noise_dim)
        print("od_dim", od_dim)
        self.fc1 = nn.Linear(noise_dim, 512)
        # 定义 Dropout 层，丢弃概率为 dropout_prob
        # self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, od_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.sigmoid(self.fc1(z))
        # h = self.dropout1(h)
        od = self.sigmoid(self.fc2(h))
        return od

class TODVolumeMapping(nn.Module):
    def __init__(self, od_dim, link_dim):
        super(TODVolumeMapping, self).__init__()
        n1 = 1024
        n2 = 256
        # self.fc1 = nn.Linear(od_dim, n1)
        # self.fc2 = nn.Linear(n1, n2)
        self.relu = nn.ReLU()
        # self.fc3 = nn.Linear(n2, link_dim)

        self.fc1 = nn.Linear(od_dim, n1)
        self.fc2 = nn.Linear(n1, link_dim)


        self.sigmoid = nn.Sigmoid()


    def forward(self, od):

        # h1 = self.relu(self.fc1(od))
        # h2 = self.relu(self.fc2(h1))
        # speed = self.fc3(h2)

        h1 = self.relu(self.fc1(od))
        speed = self.fc2(h1)
        return speed


class OVSModel(nn.Module):
    def __init__(self, noise_dim, od_dim, link_dim, hidden_size):
        super(OVSModel, self).__init__()
        self.tod_generator = TODGenerator(noise_dim, od_dim)
        self.tod_volume_mapping = TODVolumeMapping(od_dim, link_dim)


    def forward(self, z):
        od = self.tod_generator(z)
        speed = self.tod_volume_mapping(od)
        return od, speed


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
    

# 保存日志文件
log_filename = f"log/training_log_25.1.14版本_{time.strftime('%Y%m%d_%H%M%S')}.log"
with open(log_filename, 'w') as log_file:
    log_file.write("Epoch, Train Loss, Validation Loss\n")


def train_model(model, train_loader, val_loader, test_loader,epochs, patience, learning_rate, lr2,test_lr,ft_epochs):
    
    # 损失函数和优化器
    criterion = nn.MSELoss()

    optimizer_od_flow = optim.Adam(model.tod_volume_mapping.parameters(), lr=lr2)
    optimizer_od_gen = optim.Adam(model.tod_generator.parameters(), lr=test_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    best_val_loss = float('inf')
    patience_counter = 0


    # Stage 1
    print("Training TOD-Volume Mapping...")

    # 固定 TOD-Volume Mapping 和 Volume-Speed Mapping 参数
    for param in model.tod_volume_mapping.parameters():
        param.requires_grad = True
    for param in model.tod_generator.parameters():
        param.requires_grad = False
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0
        for od, _, speed in train_loader:
            od,speed = od.to(device), speed.to(device)
            # 训练 TOD-Volume Mapping
            optimizer_od_flow.zero_grad()
            speed_pred = model.tod_volume_mapping(od)
            loss = criterion(speed_pred, speed)
            loss.backward()
            optimizer_od_flow.step()

            train_loss += loss.item()

        # 计算训练集的平均损失
        train_loss /= len(train_loader)

        # 验证过程
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, od, _, speed in val_loader:
                od,speed = od.to(device), speed.to(device)
                speed_pred = model.tod_volume_mapping(od)
                loss = criterion(speed_pred, speed)

                val_loss += loss.item()

        # 计算验证集的平均损失
        val_loss /= len(val_loader)

        # 保存每一轮的损失，并打印
        with open(log_filename, 'a') as log_file:
            log_file.write(f"[2]{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}\n")

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # ({optimizer.param_groups[0]['lr']:.6f})

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "ckpt/best_model_feature1_25.1.14数据集版本_stage2.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1


        # 早停 注释
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break




    # 测试阶段
    # Stage 3
    print("Testing TOD Generation...")

    model.load_state_dict(torch.load("ckpt/best_model_feature1_25.1.14数据集版本_stage2.pth"))
    best_val_loss = float('inf')
    patience_counter = 0

    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)


    final_od_pred = None
    loss_total = 0
    mae_total = 0
    rmse_total = 0
    all_real_od = []
    all_pred_od = []


    # 固定 TOD-Volume Mapping 和 Volume-Speed Mapping 参数
    patience = 100
    for param in model.tod_volume_mapping.parameters():
        param.requires_grad = False
    for param in model.tod_generator.parameters():
        param.requires_grad = True

    # 微调 TOD Generation
    for epoch in range(ft_epochs):
        model.train()
        train_loss = 0
        # print("---------------------")
        for z_test, od_test, _, speed_test in test_loader:
            z_test,od_test,speed_test = z_test.to(device),od_test.to(device),speed_test.to(device)
            optimizer_od_gen.zero_grad()
            od_pred = model.tod_generator(z_test)
            # print("OD总和",torch.sum(od_pred))
            speed_pred = model.tod_volume_mapping(od_pred)
            loss = criterion(speed_pred, speed_test)
            loss.backward()
            optimizer_od_gen.step()
            train_loss += loss.item()
            # if epoch % 20 == 0:
            #     print(f"batch内的OD预测和标签及误差：\n{od_pred[:3, :6]}\n{od_test[:3, :6]},{criterion(od_pred, od_test)}")

        train_loss /= len(test_loader)
        # print(f"Epoch [{epoch + 1}/{ft_epochs}], Train Loss: {train_loss:.4f}")

        # 验证过程
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z_val, od, _, speed in val_loader:
                z_val,od,speed = z_val.to(device),od.to(device), speed.to(device)
                od_pred = model.tod_generator(z_val)
                speed_pred = model.tod_volume_mapping(od_pred)
                loss = criterion(speed_pred, speed)

                val_loss += loss.item()

        # 计算验证集的平均损失
        val_loss /= len(val_loader)

        # 保存每一轮的损失，并打印
        # with open(log_filename, 'a') as log_file:
        #     log_file.write(f"[2]{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}\n")

        print(f"Epoch [{epoch + 1}/{ft_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # ({optimizer.param_groups[0]['lr']:.6f})

        # 提前停止机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            # torch.save(model.state_dict(), "ckpt/best_model_feature1_25.1.14数据集版本_stage3.pth")
            # print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1


        # 早停 注释
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


    print("微调完成，模型已保存")
    torch.save(model.state_dict(), "ckpt/best_model_feature1_25.1.14数据集版本_stage3.pth")

    print("加载微调后模型，进行推理")
    model.load_state_dict(torch.load("ckpt/best_model_feature1_25.1.14数据集版本_stage3.pth"))
    model.eval()

    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0

    test_loss_speed = 0
    rmse_total_speed = 0
    mae_total_speed = 0
    mape_total_speed = 0

    all_real_od = []
    all_pred_od = []
    with torch.no_grad():
        # 最终重建的 TOD
        for z_test, od_test, _, speed_test in test_loader:
            z_test, od_test, speed_test = z_test.to(device), od_test.to(device), speed_test.to(device)

            od_pred = model.tod_generator(z_test)
            speed_pred = model.tod_volume_mapping(od_pred)

            loss1 = criterion(od_pred, od_test)
            loss2 = criterion(speed_pred, speed_test)
            test_loss += loss1.item()
            test_loss_speed += loss2.item()

            # 计算 RMSE 和 MAE
            rmse1, mae1, mape1 = calculate_rmse_mae(od_pred, od_test)
            rmse2, mae2, mape2 = calculate_rmse_mae(speed_pred, speed_test)
            # print(f"batch内的OD预测和标签：\n{od_pred[:3,:6]}\n{od_test[:3,:6]}")
            # print(f"batch内的OD预测和标签及误差：\n{od_pred[:3, :6]}\n{od_test[:3, :6]},{loss}")

            # print(f"batch的Loss,RMSE和MAE:{loss:.4f},{rmse:.4f},{mae:.4f}")
            rmse_total += rmse1
            mae_total += mae1
            mape_total += mape1

            rmse_total_speed += rmse2
            mae_total_speed += mae2
            mape_total_speed += mape2

            all_real_od.append(od_test.cpu().detach().numpy())
            all_pred_od.append(od_pred.cpu().detach().numpy())


    print(f"总RMSE和MAE：{rmse_total:.4f},{mae_total:.4f}")
    print(f"测试集大小：{len(test_loader)}")
    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)

    test_loss_speed /= len(test_loader)
    rmse_total_speed /= len(test_loader)
    mae_total_speed /= len(test_loader)
    mape_total_speed /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {rmse_total:.4f} Test MAE: {mae_total:.4f} Test MAPE: {mape_total:.4f}")
    print(f"Test Speed Loss: {test_loss_speed:.4f}")
    print(f"Test Speed RMSE: {rmse_total_speed:.4f} Test Speed MAE: {mae_total_speed:.4f} Test Speed MAPE: {mape_total_speed:.4f}")

    torch.save(model.state_dict(),
               f"ckpt/best_model_feature1_25.1.14数据集版本_{test_loss:.4f}_{rmse_total:.4f}_{mae_total:.4f}_lr_{test_lr}_stage3.pth")

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
    noise_dim = 50
    od_dim = N
    link_dim = L
    hidden_size = 128
    epochs = 2000
    ft_epochs = 6000

    # 初始化模型
    model = OVSModel(noise_dim, od_dim, link_dim, hidden_size)

    lr = 0.002 # best 0.002 0.001 0.01
    lr2 = 0.001
    test_lr = 0.01
    train_model(model, train_loader, val_loader,test_loader, epochs=epochs,
                patience=20, learning_rate=lr,lr2 =lr2,test_lr=test_lr,ft_epochs=ft_epochs)



    # # 测试模型
    # test_model(model, test_loader,z,epochs=200, patience=20, learning_rate=lr)


if __name__ == "__main__":
    main()
