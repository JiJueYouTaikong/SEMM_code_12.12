import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns


# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Transformer-based OD Model
class ODModel(nn.Module):
    def __init__(self, N, F, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super(ODModel, self).__init__()
        self.N = N
        self.F = F
        self.d_model = d_model

        # Input embedding layer
        self.input_embedding = nn.Linear(F, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, N)  # Output N values for each position
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: [batch_size, N, F]
        batch_size = x.size(0)

        # Input embedding
        x = self.input_embedding(x)  # [batch_size, N, d_model]

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, N, d_model]

        # Project to output space
        x = self.output_projection(x)  # [batch_size, N, N]

        # Reshape to OD matrix
        od_matrix = x.view(batch_size, self.N, self.N)

        return od_matrix


# Data loading and preprocessing
def load_data():
    set_seed(42)

    # Load data (replace with your actual data paths)
    speed = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy')
    od = np.load('../data/OD_完整批处理_3.17_Final_MCM_60.npy')

    T, N = speed.shape

    # Generate random features
    x_random = np.random.normal(loc=0.05, scale=0.01, size=(T, N))
    x_random = np.clip(x_random, 0, 0.1)

    # Split data (60-20-20)
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)
    train_size = int(T * 0.9537)
    val_size = int(T * 0.0225)

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # Temporal features
    od_train_departures = np.sum(od_train, axis=-1)
    mean_speed = np.mean(speed_train, axis=0)
    mean_departures = np.mean(od_train_departures, axis=0)
    temporal = mean_departures - mean_speed

    temporal_expanded_train = np.tile(temporal, (speed_train.shape[0], 1))
    temporal_expanded_val = np.tile(temporal, (speed_val.shape[0], 1))
    temporal_expanded_test = np.tile(temporal, (speed_test.shape[0], 1))

    # Frequency features
    speed_freq = np.load('../data/速度的周期状态_对应25.1.14的速度数据集.npy')
    od_freq = np.load('../data/OD的周期状态_对应25.1.14的OD数据集.npy')
    freq = od_freq - speed_freq

    freq_expanded_train = np.tile(freq, (speed_train.shape[0], 1))
    freq_expanded_val = np.tile(freq, (speed_val.shape[0], 1))
    freq_expanded_test = np.tile(freq, (speed_test.shape[0], 1))

    # Combine features
    x_train = np.stack([speed_train, temporal_expanded_train, freq_expanded_train, x_random_train], axis=-1)
    x_val = np.stack([speed_val, temporal_expanded_val, freq_expanded_val, x_random_val], axis=-1)
    x_test = np.stack([speed_test, temporal_expanded_test, freq_expanded_test, x_random_test], axis=-1)

    # Normalize
    scaler = MinMaxScaler()
    train_3 = x_train[..., :3]
    val_3 = x_val[..., :3]
    test_3 = x_test[..., :3]

    x_train[..., :3] = scaler.fit_transform(train_3.reshape(-1, 3)).reshape(train_3.shape)
    x_val[..., :3] = scaler.transform(val_3.reshape(-1, 3)).reshape(val_3.shape)
    x_test[..., :3] = scaler.transform(test_3.reshape(-1, 3)).reshape(test_3.shape)

    # Use only speed feature for transformer input
    train_data = x_train[..., 0:1]  # Using only speed feature (N, 1)
    val_data = x_val[..., 0:1]
    test_data = x_test[..., 0:1]

    # Convert to tensors
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

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


# Training function
def train_model(model, train_loader, val_loader, epochs=100, patience=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ckpt/best_transformer_model.pth")
            print(f"best saved at epoch{epoch + 1},best：{best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


# Evaluation metrics
def calculate_rmse_mae(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - targets))

    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = torch.mean(torch.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]))
    else:
        mape = torch.tensor(0.0)
    return rmse.item(), mae.item(), mape.item()


# Testing function
def test_model(model, test_loader,log_filename,lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load("ckpt/best_transformer_model.pth"))

    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    rmse_total = 0
    mae_total = 0
    mape_total = 0

    all_real_od = []
    all_pred_od = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Create mask for diagonal elements
            mask = torch.ones_like(targets)
            for i in range(targets.size(1)):
                mask[:, i, i] = 0

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            rmse, mae, mape = calculate_rmse_mae(outputs * mask, targets * mask)
            rmse_total += rmse
            mae_total += mae
            mape_total += mape

            all_real_od.append(targets.cpu().numpy())
            all_pred_od.append(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    rmse_total /= len(test_loader)
    mae_total /= len(test_loader)
    mape_total /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"RMSE: {rmse_total:.4f}, MAE: {mae_total:.4f}, MAPE: {mape_total:.4f}")
    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"Lr = {lr},Test Loss: {test_loss:.4f} RMSE: {rmse_total:.4f} MAE: {mae_total:.4f} MAPE: {mape_total:.4f}\n")

    # Visualize results
    all_real_od = np.concatenate(all_real_od, axis=0)
    all_pred_od = np.concatenate(all_pred_od, axis=0)
    avg_real_od = np.mean(all_real_od, axis=0)
    avg_pred_od = np.mean(all_pred_od, axis=0)

    # Plot heatmaps
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # sns.heatmap(avg_real_od, cmap="Blues")
    # plt.title("Average True OD Matrix")
    #
    # plt.subplot(1, 2, 2)
    # sns.heatmap(avg_pred_od, cmap="Blues")
    # plt.title("Average Predicted OD Matrix")
    # plt.show()

    vmin = min(all_real_od[-2].min(), all_pred_od[-2].min())
    vmax = max(all_real_od[-2].max(), all_pred_od[-2].max())

    true_max = all_real_od[-2].max()
    pred_max = all_pred_od[-2].max()
    print(f"真实最大值{true_max},预测最大值，{pred_max}")

    # Plot heatmaps
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(all_real_od[-2], cmap="Blues", cbar=True, vmin=vmin, vmax=378)
    plt.title("True OD Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(all_pred_od[-2], cmap="Blues", cbar=True, vmin=vmin, vmax=378)
    plt.title("Predicted OD Matrix")
    plt.savefig("Transformer_prediction.png")
    # plt.show()


    return test_loss, rmse_total, mae_total, mape_total


# Main function
def main():
    # 保存日志文件
    log_filename = f"log/Transformer_MCM_完整批处理调参.log"

    # 定义学习率列表
    # lr_list = [0.01, 0.005, 0.004, 0.003, 0.002, 0.001,
    #            0.0005,0.0002,0.0001,0.00005,0.00002,0.00001]
    lr_list = [ 0.01, 0.005,0.001, 0.0005, 0.0002, 0.0001,0.00005,0.00002,0.00001]
    # lr_list = [ 0.0005]  # 是否当前最佳 YES 6.20


    # 遍历学习率列表
    for lr in lr_list:
        print(f"当前学习率: {lr}")

        set_seed(42)

        # Load data
        train_loader, val_loader, test_loader = load_data()

        # Initialize model
        model = ODModel(N=110, F=1, d_model=256, nhead=8, num_layers=3)

        # Train model
        train_model(model, train_loader, val_loader, epochs=2000, patience=20, learning_rate=lr)

        # Test model
        test_loss, rmse, mae, mape = test_model(model, test_loader,log_filename,lr)

        print(f"Final Results - Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")


if __name__ == "__main__":
    main()