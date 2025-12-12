import numpy as np
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime

# ==============================================================
# 1️⃣ 随机种子与配置
# ==============================================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# ==============================================================
# 2️⃣ 日志系统
# ==============================================================
def init_log():
    os.makedirs("log", exist_ok=True)
    log_name = f"log/RandomForest_OD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return log_name

def write_log(log_file, msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# ==============================================================
# 3️⃣ 数据加载与预处理
# ==============================================================
def load_data(is_mcm=True):
    set_seed(42)

    if is_mcm:
        speed = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final_MCM_60.npy')
        test_size = 35
        val_size = 33
        T, N = speed.shape
        train_size = T - test_size - val_size
    else:
        speed = np.load('../data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final.npy')
        T, N = speed.shape
        train_size = int(T * 0.6)
        val_size = int(T * 0.2)

    # 顺序划分
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    X_train, X_val, X_test = speed[train_indices], speed[val_indices], speed[test_indices]
    Y_train, Y_val, Y_test = od[train_indices], od[val_indices], od[test_indices]

    print("Train:", X_train.shape, Y_train.shape)
    print("Val:", X_val.shape, Y_val.shape)
    print("Test:", X_test.shape, Y_test.shape)

    # 归一化输入特征
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 输出扁平化为 (样本数, N*N)
    Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)
    Y_val_flat = Y_val.reshape(Y_val.shape[0], -1)
    Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)

    return X_train_scaled, X_val_scaled, X_test_scaled, Y_train_flat, Y_val_flat, Y_test_flat, N

# ==============================================================
# 4️⃣ 计算指标函数
# ==============================================================
def calculate_metrics(y_pred, y_true):
    y_pred_torch = torch.tensor(y_pred)
    y_true_torch = torch.tensor(y_true)

    T = y_pred_torch.shape[0]
    N = int(np.sqrt(y_pred_torch.shape[1]))

    y_pred_torch = y_pred_torch.view(T, N, N)
    y_true_torch = y_true_torch.view(T, N, N)

    mse = torch.mean((y_pred_torch - y_true_torch) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_pred_torch - y_true_torch))

    non_zero_mask = y_true_torch != 0
    mape = torch.mean(torch.abs((y_pred_torch[non_zero_mask] - y_true_torch[non_zero_mask]) / y_true_torch[non_zero_mask]))

    # CPC
    pred_flat = y_pred_torch.reshape(T, -1)
    targ_flat = y_true_torch.reshape(T, -1)
    cpc_list = []
    for t in range(T):
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
    min_val = 1e-8
    for t in range(T):
        pred_t = pred_flat[t]
        targ_t = targ_flat[t]
        pred_dist = (pred_t + min_val) / (torch.sum(pred_t) + min_val * pred_t.numel())
        targ_dist = (targ_t + min_val) / (torch.sum(targ_t) + min_val * targ_t.numel())
        pred_dist = torch.clamp(pred_dist, min=min_val)
        targ_dist = torch.clamp(targ_dist, min=min_val)
        m = 0.5 * (pred_dist + targ_dist)
        kl1 = torch.sum(pred_dist * torch.log(pred_dist / m))
        kl2 = torch.sum(targ_dist * torch.log(targ_dist / m))
        jsd_t = 0.5 * (kl1 + kl2)
        if not torch.isnan(jsd_t):
            jsd_list.append(jsd_t.item())
    jsd = sum(jsd_list) / len(jsd_list)

    return rmse.item(), mae.item(), mape.item(), cpc, jsd

# ==============================================================
# 5️⃣ 训练模型
# ==============================================================
def train_rf(X_train, Y_train, X_val, Y_val, params, log_file):
    write_log(log_file, f"=== 训练参数 ===\n{params}\n")

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, Y_train)

    # 验证集 MSE
    Y_val_pred = rf.predict(X_val)
    mse_val = mean_squared_error(Y_val, Y_val_pred)
    write_log(log_file, f"Validation MSE: {mse_val:.4f}")

    return rf

# ==============================================================
# 6️⃣ 测试模型
# ==============================================================
def test_rf(model, X_test, Y_test, log_file):
    Y_pred = model.predict(X_test)
    rmse, mae, mape, cpc, jsd = calculate_metrics(Y_pred, Y_test)

    write_log(log_file, "\n=== 测试集评估结果 ===")
    write_log(log_file, f"Test RMSE: {rmse:.4f}")
    write_log(log_file, f"Test MAE:  {mae:.4f}")
    write_log(log_file, f"Test MAPE: {mape:.4f}")
    write_log(log_file, f"Test CPC:  {cpc:.4f}")
    write_log(log_file, f"Test JSD:  {jsd:.4f}")

# ==============================================================
# 7️⃣ 主程序
# ==============================================================
def main():
    log_file = init_log()
    write_log(log_file, "✅ RandomForest 多输出 OD 估计任务启动")

    X_train, X_val, X_test, Y_train, Y_val, Y_test, N = load_data(is_mcm=True)

    params = {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }

    model = train_rf(X_train, Y_train, X_val, Y_val, params, log_file)
    test_rf(model, X_test, Y_test, log_file)

    write_log(log_file, "\n✅ 任务完成\n")

if __name__ == "__main__":
    main()
