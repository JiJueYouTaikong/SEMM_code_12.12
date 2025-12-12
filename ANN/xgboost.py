import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import os

# ==== 评价函数（基本保留，只在使用时把 numpy 转为 torch） ====
def calculate_rmse_mae(predictions, targets):
    '''
    :param predictions: torch.Tensor [T,N,N]
    :param targets: torch.Tensor [T,N,N]
    :return: rmse, mae, mape, cpc, jsd
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

    return rmse.item(), mae.item(), mape.item(), cpc, jsd


# ==== 随机种子 ====
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==== 数据加载（返回 numpy 矩阵形式以供 XGBoost 使用） ====
def load_data(is_mcm):
    set_seed(42)

    if is_mcm:
        speed = np.load('../data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final_MCM_60.npy')

        # 获取数据长度 T
        T, N = speed.shape
        # 根据指定的时间步数划分
        test_size = 35
        val_size = 33
        train_size = T - test_size - val_size

    else:
        speed = np.load('../data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('../data/OD_完整批处理_3.17_Final.npy')

        T, N = speed.shape
        train_size = int(T * 0.6)
        val_size = int(T * 0.2)

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]

    print("6:2:2顺序划分的训练集Speed", speed_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集Speed", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集Speed", speed_test.shape, "OD", od_test.shape)

    scaler = MinMaxScaler()
    x_train_scale = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    x_val_scale = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    x_test_scale = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)

    # 准备 X (T,N) 和 y (T, N*N)
    # od_train/val/test 原始是 [T, N, N]，需要 flatten 为 [T, N*N]
    Ttr, N = x_train_scale.shape
    y_train = od_train.reshape(od_train.shape[0], -1)
    y_val = od_val.reshape(od_val.shape[0], -1)
    y_test = od_test.reshape(od_test.shape[0], -1)

    X_train = x_train_scale  # shape [Ttr, N]
    X_val = x_val_scale
    X_test = x_test_scale

    print("归一化并展平后的形状：",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_val", X_val.shape, "y_val", y_val.shape,
          "X_test", X_test.shape, "y_test", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, N


# ==== 训练 XGBoost 多输出模型 ====
def train_xgb_multioutput(X_train, y_train, X_val=None, y_val=None,
                          model_path="ckpt/xgb_multi.pkl",
                          n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=8, random_state=42):
    """
    使用 MultiOutputRegressor 封装 XGBRegressor 训练多输出回归
    注意：当输出维非常大（N*N）时训练耗时/内存会显著上升。
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    xgb = XGBRegressor(objective='reg:squarederror',
                       n_estimators=n_estimators,
                       learning_rate=learning_rate,
                       max_depth=max_depth,
                       verbosity=0,
                       n_jobs=n_jobs,
                       random_state=random_state)

    mor = MultiOutputRegressor(xgb, n_jobs=1)  # 内部并行会占用资源，n_jobs=1 保守设置

    print("开始训练 XGBoost 多输出模型，输出维度:", y_train.shape[1])
    mor.fit(X_train, y_train)  # 注意：没有 per-output early stopping（如需可改为逐列训练）

    joblib.dump(mor, model_path)
    print("模型已保存到:", model_path)
    return mor


# ==== 测试/评估（加载模型并计算你的指标） ====
log_filename = f"log/XGB_训练日志.log"

def test_xgb_model(model, X_test, y_test, N, model_path=None):
    """
    model: 如果为 None，则从 model_path 加载
    X_test: numpy [Ttest, N]
    y_test: numpy [Ttest, N*N]
    N: 区域数量
    """
    if model is None:
        assert model_path is not None, "需要 model_path 或已传入 model"
        model = joblib.load(model_path)

    preds = model.predict(X_test)  # shape [Ttest, N*N]
    # 如果 XGBoost 输出有负值（可能），根据业务需要裁剪到 0
    preds = np.clip(preds, a_min=0.0, a_max=None)

    # reshape 为 [T, N, N]
    preds_3d = preds.reshape(preds.shape[0], N, N)
    real_3d = y_test.reshape(y_test.shape[0], N, N)

    # mask 对角线为 0（参照你原有实现）
    mask = np.ones_like(real_3d)
    for i in range(N):
        mask[:, i, i] = 0

    # 转为 torch.Tensor 以复用 calculate_rmse_mae
    preds_t = torch.tensor(preds_3d * mask, dtype=torch.float32)
    real_t = torch.tensor(real_3d, dtype=torch.float32)

    rmse, mae, mape, cpc, jsd = calculate_rmse_mae(preds_t, real_t)

    with open(log_filename, 'a') as log_file:
        log_file.write(f"XGB Test RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}, CPC: {cpc:.6f}, JSD: {jsd:.6f}\n")

    print(f"XGB Test RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}, CPC: {cpc:.6f}, JSD: {jsd:.6f}")
    return preds_3d, real_3d


# ==== 主流程 ====
def main():
    X_train, y_train, X_val, y_val, X_test, y_test, N = load_data(is_mcm=False)

    # 超参数（可调整）
    xgb_params = dict(
        n_estimators=50,
        learning_rate=0.03,
        max_depth=3,
        n_jobs=8,
        random_state=42
    )

    model_path = "ckpt/xgb_multi.pkl"

    # 训练（若已存在模型文件，可直接 load）
    if os.path.exists(model_path):
        print("检测到已存在模型，直接加载：", model_path)
        model = joblib.load(model_path)
    else:
        model = train_xgb_multioutput(X_train, y_train, X_val, y_val,
                                      model_path=model_path,
                                      n_estimators=xgb_params['n_estimators'],
                                      learning_rate=xgb_params['learning_rate'],
                                      max_depth=xgb_params['max_depth'],
                                      n_jobs=xgb_params['n_jobs'],
                                      random_state=xgb_params['random_state'])

    # 测试
    preds_3d, real_3d = test_xgb_model(model, X_test, y_test, N, model_path=model_path)

if __name__ == "__main__":
    main()
