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
def init_log(is_mcm):
    os.makedirs("log", exist_ok=True)
    log_name = f"log/RandomForest_{is_mcm}.log"
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

    return rf,mse_val

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

    Y_pred_re = Y_pred.reshape(-1,110,110)
    return Y_pred_re


# ==============================================================
# 主程序：对 is_mcm=True/False 分别使用预设最佳参数训练、测试并保存结果
# ==============================================================
def main():
    # 两组预设的“最佳参数”（直接使用，不做网格搜索）
    best_params_by_flag = {
        True: {
            "n_estimators": 50,
            "max_depth": 15,
            "min_samples_leaf": 10
        },
        False: {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 1
        }
    }

    for flag in (True, False):
        is_mcm = flag
        log_file = init_log(is_mcm)
        write_log(log_file, f"\n\n===== 开始运行 is_mcm={is_mcm} 时间: {datetime.now()} =====")

        X_train, X_val, X_test, Y_train, Y_val, Y_test, N = load_data(is_mcm=is_mcm)

        # 取出对应的最佳参数并补全常用字段
        params = best_params_by_flag[is_mcm].copy()
        params["n_jobs"] = -1
        params["random_state"] = 42

        write_log(log_file, f"使用直接指定的最佳参数（不网格搜索）: {params}")

        # 训练与验证
        model, mse_val = train_rf(X_train, Y_train, X_val, Y_val, params, log_file)

        # 测试并返回预测
        Y_pred = test_rf(model, X_test, Y_test, log_file)

        # 打印测试预测结果维度
        print(f"is_mcm={is_mcm} -> Y_pred shape: {Y_pred.shape}")
        print(f"is_mcm={is_mcm} -> Y_test shape: {Y_test.shape}")

        # 保存预测和真实标签为 .npy 文件（文件名包含 is_mcm 标识和时间戳以防覆盖）
        suf = "True" if is_mcm else "False"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_fname = f"predictions_is_mcm_{suf}_{ts}.npy"

        np.save(pred_fname, Y_pred)
        write_log(log_file, f"Saved predictions -> {pred_fname}")

    print("全部运行结束。")

if __name__ == "__main__":
    main()



# # ==============================================================
# # 7️⃣ 主程序   网格搜索版本
# # ==============================================================
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np
#
#
# def main():
#     is_mcm = True
#     log_file = init_log(is_mcm)
#
#     X_train, X_val, X_test, Y_train, Y_val, Y_test, N = load_data(is_mcm=is_mcm)
#
#     # 定义网格搜索参数
#     param_grid = {
#         "n_estimators": [50, 100,200],
#         "max_depth": [3, 5, 7, 10,15],
#         "min_samples_leaf": [1, 3, 5,10,15]
#     }
#
#
#     # true
#     param_grid = {
#         "n_estimators": [50],
#         "max_depth": [15],
#         "min_samples_leaf": [10]
#     }
#
#     # false
#     param_grid = {
#         "n_estimators": [100],
#         "max_depth": [5],
#         "min_samples_leaf": [1]
#     }
#
#     # 执行网格搜索
#     best_params, best_mse = grid_search_rf(
#         X_train, Y_train, X_val, Y_val, param_grid, log_file
#     )
#
#     # 使用最佳参数训练最终模型
#     best_params["n_jobs"] = -1
#     best_params["random_state"] = 42
#
#     print(f"最佳参数: {best_params}")
#     print(f"最佳验证集MSE: {best_mse:.4f}")
#
#     # # 记录到日志文件
#     # log_file.write(f"最佳参数: {best_params}\n")
#     # log_file.write(f"最佳验证集MSE: {best_mse:.4f}\n")
#
#     # 使用最佳参数训练模型
#     model, mse_val = train_rf(X_train, Y_train, X_val, Y_val, best_params, log_file)
#     test_rf(model, X_test, Y_test, log_file)
#
#
# def grid_search_rf(X_train, Y_train, X_val, Y_val, param_grid, log_file=None):
#     """
#     执行随机森林的网格搜索
#     """
#     best_score = float('inf')
#     best_params = None
#
#     # 生成所有参数组合
#     n_estimators_list = param_grid["n_estimators"]
#     max_depth_list = param_grid["max_depth"]
#     min_samples_leaf_list = param_grid["min_samples_leaf"]
#
#     total_combinations = len(n_estimators_list) * len(max_depth_list) * len(min_samples_leaf_list)
#     current_combination = 0
#
#     print(f"开始网格搜索，共 {total_combinations} 种参数组合...")
#
#     for n_estimators in n_estimators_list:
#         for max_depth in max_depth_list:
#             for min_samples_leaf in min_samples_leaf_list:
#                 current_combination += 1
#
#                 # 设置参数
#                 params = {
#                     "n_estimators": n_estimators,
#                     "max_depth": max_depth,
#                     "min_samples_leaf": min_samples_leaf,
#                     "n_jobs": -1,
#                     "random_state": 42
#                 }
#
#                 print(f"正在训练 [{current_combination}/{total_combinations}]: {params}")
#
#                 # 训练模型
#                 model = RandomForestRegressor(**params)
#                 model.fit(X_train, Y_train)
#
#                 # 在验证集上评估
#                 y_val_pred = model.predict(X_val)
#                 mse = mean_squared_error(Y_val, y_val_pred)
#
#                 # 更新最佳结果
#                 if mse < best_score:
#                     best_score = mse
#                     best_params = params.copy()
#
#                 print(f"  MSE: {mse:.4f} | 最佳MSE: {best_score:.4f}")
#
#     print(f"网格搜索完成！最佳参数: {best_params}, 最佳MSE: {best_score:.4f}")
#
#     # 记录到日志文件
#     if log_file:
#         write_log(log_file,f"网格搜索完成！共尝试 {total_combinations} 种参数组合\n")
#         write_log(log_file,f"最佳参数: {best_params}\n")
#         write_log(log_file,f"最佳验证集MSE: {best_score:.4f}\n")
#
#     return best_params, best_score


