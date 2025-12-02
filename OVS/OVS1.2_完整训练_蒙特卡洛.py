import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import get_data,get_60days_data
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
        self.fc2 = nn.Linear(512, od_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.sigmoid(self.fc1(z))
        od = self.sigmoid(self.fc2(h))
        return od


class TODVolumeMapping(nn.Module):
    def __init__(self, od_dim, link_dim):
        super(TODVolumeMapping, self).__init__()
        self.fc_route = nn.Linear(od_dim, link_dim)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.fc_att = nn.Linear(64, link_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, od):
        # print("----------------------------")
        # p: torch.Size([32, 762])
        # p: torch.Size([32, 1, 762])
        # h1: torch.Size([32, 32, 760])
        # ek: torch.Size([32, 64, 758])
        # e: torch.Size([32, 64])
        # alpha: torch.Size([32, 762])
        # alpha: torch.Size([32, 762, 1])
        # flow: torch.Size([32, 762])
        # print("od:", od.shape)

        p = self.sigmoid(self.fc_route(od))  # [T,Link]
        # print(f"p:{p.shape}")

        p = p.unsqueeze(1)  # [T,1,Link]
        # print(f"p:{p.shape}")

        h1 = self.conv1(p)
        # print(f"h1:{h1.shape}")

        ek = self.conv2(h1)
        # print(f"ek:{ek.shape}")

        e = ek.mean(dim=2)
        # print(f"e:{e.shape}")

        alpha = self.softmax(self.fc_att(e))
        # print(f"alpha:{alpha.shape}")

        alpha = alpha.unsqueeze(2)
        # print(f"alpha:{alpha.shape}")

        flow = torch.sum(alpha * p, dim=1)
        # print(f"flow:{flow.shape}")

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


def train_model(model, train_loader, val_loader, test_loader, epochs, patience, learning_rate, lr2, test_lr, ft_epochs):
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_flow_speed = optim.Adam(model.volume_speed_mapping.parameters(), lr=learning_rate)
    optimizer_od_flow = optim.Adam(model.tod_volume_mapping.parameters(), lr=lr2)
    optimizer_od_gen = optim.Adam(model.tod_generator.parameters(), lr=test_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    best_val_loss = float('inf')
    patience_counter = 0

    # 训练阶段
    print("Training Volume-Speed Mapping...")

    # # 固定 TOD-Volume Mapping 和 Volume-Speed Mapping 参数
    # for param in model.tod_volume_mapping.parameters():
    #     param.requires_grad = False
    # for param in model.volume_speed_mapping.parameters():
    #     param.requires_grad = True
    # for param in model.tod_generator.parameters():
    #     param.requires_grad = False
    #
    # for epoch in range(epochs):
    #     model.train()
    #     train_loss = 0
    #     for _, flow, speed in train_loader:
    #         flow,speed = flow.to(device), speed.to(device)
    #         # 训练 Volume-Speed Mapping
    #         optimizer_flow_speed.zero_grad()
    #         speed_pred = model.volume_speed_mapping(flow)
    #         loss = criterion(speed_pred, speed)
    #         loss.backward()
    #         optimizer_flow_speed.step()
    #
    #         train_loss += loss.item()
    #
    #     train_loss /= len(train_loader)
    #
    #     # 验证过程
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for _,_, flow, speed in val_loader:
    #             flow,speed = flow.to(device), speed.to(device)
    #             speed_pred = model.volume_speed_mapping(flow)
    #             loss = criterion(speed_pred, speed)
    #             val_loss += loss.item()
    #
    #     # 计算验证集的平均损失
    #     val_loss /= len(val_loader)
    #
    #     # 保存每一轮的损失，并打印
    #     with open(log_filename, 'a') as log_file:
    #         log_file.write(f"[1]{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}\n")
    #
    #     print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    #     # ({optimizer.param_groups[0]['lr']:.6f})
    #
    #     # 提前停止机制
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
    #         # 保存最佳模型
    #         torch.save(model.state_dict(), "ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage1.pth")
    #         print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
    #     else:
    #         patience_counter += 1
    #
    #     # 早停 注释
    #     if patience_counter >= patience:
    #         print("Early stopping triggered.")
    #         break

    # 绘制训练过程中的损失曲线
    # with open(log_filename, 'r') as log_file:
    #     epochs_list, train_loss_list, val_loss_list = [], [], []
    #     for line in log_file.readlines()[1:]:
    #         epoch, train_loss, val_loss = line.strip().split(", ")
    #         epochs_list.append(int(epoch))
    #         train_loss_list.append(float(train_loss))
    #         val_loss_list.append(float(val_loss))
    #
    # plt.plot(epochs_list, train_loss_list, label='Train Loss')
    # plt.plot(epochs_list, val_loss_list, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title(f'Stage1: train and validation loss with lr={learning_rate}')
    # plt.savefig("loss_curve.png")
    # plt.show()

    # Stage 2
    print("Training TOD-Volume Mapping...")

    # # 固定 TOD-Volume Mapping 和 Volume-Speed Mapping 参数
    # for param in model.tod_volume_mapping.parameters():
    #     param.requires_grad = True
    # for param in model.volume_speed_mapping.parameters():
    #     param.requires_grad = False
    # for param in model.tod_generator.parameters():
    #     param.requires_grad = False
    #
    # model.load_state_dict(torch.load("ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage1.pth"))
    # best_val_loss = float('inf')
    # patience_counter = 0
    #
    # for epoch in range(int(epochs)):
    #     model.train()
    #     train_loss = 0
    #     for od, _, speed in train_loader:
    #         od,speed = od.to(device), speed.to(device)
    #         # 训练 TOD-Volume Mapping
    #         optimizer_od_flow.zero_grad()
    #         flow_pred = model.tod_volume_mapping(od)
    #         speed_pred = model.volume_speed_mapping(flow_pred)
    #         loss = criterion(speed_pred, speed)
    #         loss.backward()
    #         optimizer_od_flow.step()
    #
    #         train_loss += loss.item()
    #
    #     # 计算训练集的平均损失
    #     train_loss /= len(train_loader)
    #
    #     # 验证过程
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for _, od, _, speed in val_loader:
    #             od,speed = od.to(device), speed.to(device)
    #             flow_pred = model.tod_volume_mapping(od)
    #             speed_pred = model.volume_speed_mapping(flow_pred)
    #             loss = criterion(speed_pred, speed)
    #
    #             val_loss += loss.item()
    #
    #     # 计算验证集的平均损失
    #     val_loss /= len(val_loader)
    #
    #     # 保存每一轮的损失，并打印
    #     with open(log_filename, 'a') as log_file:
    #         log_file.write(f"[2]{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}\n")
    #
    #     print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    #     # ({optimizer.param_groups[0]['lr']:.6f})
    #
    #     # 提前停止机制
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
    #         # 保存最佳模型
    #         torch.save(model.state_dict(), "ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage2.pth")
    #         print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
    #     else:
    #         patience_counter += 1
    #
    #
    #     # 早停 注释
    #     if patience_counter >= patience:
    #         print("Early stopping triggered.")
    #         break

    # 测试阶段
    # Stage 3
    print("Testing TOD Generation...")

    model.load_state_dict(torch.load("ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage2.pth"))
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
    for param in model.tod_volume_mapping.parameters():
        param.requires_grad = False
    for param in model.volume_speed_mapping.parameters():
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
            flow_pred = model.tod_volume_mapping(od_pred)
            speed_pred = model.volume_speed_mapping(flow_pred)
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
                flow_pred = model.tod_volume_mapping(od_pred)
                speed_pred = model.volume_speed_mapping(flow_pred)
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
            torch.save(model.state_dict(), "ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage3.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1


        # 早停 注释
        # if patience_counter >= patience:
        #     print("Early stopping triggered.")
        #     break

    print("微调完成，模型已保存")

    print("加载微调后模型，进行推理")
    model.load_state_dict(torch.load("ckpt_仿真/best_model_feature1_25.1.14数据集版本_stage3.pth"))
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
            flow_pred = model.tod_volume_mapping(od_pred)
            speed_pred = model.volume_speed_mapping(flow_pred)

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
    # print(f"Test Speed Loss: {test_loss_speed:.4f}")
    # print(f"Test Speed RMSE: {rmse_total_speed:.4f} Test Speed MAE: {mae_total_speed:.4f} Test Speed MAPE: {mape_total_speed:.4f}")

    torch.save(model.state_dict(),
               f"ckpt_仿真/best_model_feature1_25.1.14数据集版本_{test_loss:.4f}_{rmse_total:.4f}_{mae_total:.4f}_lr_{test_lr}_stage3.pth")

    # 计算平均的OD矩阵
    # 设置不使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    all_real_od_t = np.concatenate(all_real_od, axis=0)
    all_pred_od_t = np.concatenate(all_pred_od, axis=0)
    print(f"所有时间步的OD预测：{all_real_od_t.shape}")
    all_real_od = np.mean(all_real_od_t, axis=0)
    all_pred_od = np.mean(all_pred_od_t, axis=0)
    print(f"时间步平均后的OD预测：{all_real_od.shape}")

    all_pred = all_pred_od_t.reshape(all_real_od_t.shape[0], 110, -1)
    np.save("../可视化/测试集TNN/Pred-OVS-MCM.npy", all_pred)
    # g_reconstructed = model.tod_generator(z_test)
    # print("Reconstructed TOD shape:", g_reconstructed.shape)


# 主程序
def main():
    train_loader, val_loader, test_loader, T, N, L = get_60days_data.load_data()

    # 超参数
    noise_dim = N
    od_dim = N
    link_dim = L
    hidden_size = 128
    epochs = 600
    ft_epochs = 3000

    # 初始化模型
    model = OVSModel(noise_dim, od_dim, link_dim, hidden_size)

    lr = 0.002  # best
    lr2 = 0.001
    test_lr = 0.01
    train_model(model, train_loader, val_loader, test_loader, epochs=epochs,
                patience=20, learning_rate=lr, lr2=lr2, test_lr=test_lr, ft_epochs=ft_epochs)

    # # 测试模型
    # test_model(model, test_loader,z,epochs=200, patience=20, learning_rate=lr)


if __name__ == "__main__":
    main()
