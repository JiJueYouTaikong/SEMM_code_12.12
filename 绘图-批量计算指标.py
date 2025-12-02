import os
import numpy as np
import torch


def load_predictions():
    """加载所有预测数据，并确保'Ours'排在最后"""
    methods = {
        "DUE-LS": "./可视化/测试集TNN/Pred-DUE-LS.npy",
        "SUE-LS": "./可视化/测试集TNN/Pred-SUE-LS.npy",
        "DUE-GB": "./可视化/测试集TNN/Pred-DUE-GB.npy",
        "SUE-GB": "./可视化/测试集TNN/Pred-SUE-GB.npy",
        "PESL": "./可视化/测试集TNN/Pred-PESL.npy",
        "SSM": "./可视化/测试集TNN/Pred_SSM_SUE_MSA_6.15_7.3833.npy",
        "SSM-BO": "./可视化/测试集TNN/Pred-SSM-BO.npy",
        "SSM-LS": "./可视化/测试集TNN/Pred_SSM_LS_SUE_MSA_6.15_9.4040.npy",

        "OVS": "./可视化/测试集TNN/Pred-OVS.npy",
        "DeepGravity": "./可视化/测试集TNN/Pred_DeepGravity_6.88.npy",
        "GPT2": "./可视化/测试集TNN/Pred_GPT2.npy",
        "DS": "./可视化/测试集TNN/Pred_DeepSeek-MOE.npy",

        "OVS-MCM": "./可视化/测试集TNN/Pred-OVS-MCM.npy",
        "GPT2-MCM": "./可视化/测试集TNN/Pred_GPT2-MCM.npy",
        "DS-MCM": "./可视化/测试集TNN/Pred_DeepSeek-MOE-MCM.npy",

        "SEMM wo MCSM": "./可视化/测试集TNN/Pred_ours_ZINB.npy",
        "SEMM": "./可视化/测试集TNN/Pred_RED-5.7127.npy"

    }
    predictions = {}
    for name, path in methods.items():
        if os.path.exists(path):
            predictions[name] = np.load(path)
        else:
            print(f"警告：找不到文件 {path}，跳过该方法。")

    # 确保 Ours 始终在最后
    sorted_predictions = {}
    for key in predictions:
        if key != "Ours":
            sorted_predictions[key] = predictions[key]
    if "Ours" in predictions:
        sorted_predictions["Ours"] = predictions["Ours"]

    return sorted_predictions


def load_real_data():
    try:
        return np.load('./可视化/测试集TNN/真实值.npy')
    except FileNotFoundError:
        print("真实值文件未找到，请检查路径。")
        return None


def evaluate_metrics(predictions, targets):
    '''
    单个方法的五项指标计算：RMSE, MAE, MAPE, CPC, JSD
    '''
    predictions = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

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

        pred_t = torch.clamp(pred_flat[t], min=0)
        targ_t = torch.clamp(targ_flat[t], min=0)

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

## 原始 35样本平均
# def evaluate_all_methods():
#     predictions_dict = load_predictions()
#     real = load_real_data()
#     if real is None:
#         print("终止评估：缺少真实值。")
#         return
#
#     print(f"{'Method':<15} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'CPC':<8} {'JSD':<8}")
#     print("-" * 60)
#
#     for method, pred in predictions_dict.items():
#         rmse, mae, mape, cpc, jsd = evaluate_metrics(pred, real)
#         print(f"{method:<15} {rmse:<8.4f} {mae:<8.4f} {mape:<8.4f} {cpc:<8.4f} {jsd:<8.4f}")

def evaluate_all_methods():
    predictions_dict = load_predictions()
    real = load_real_data()
    if real is None:
        print("终止评估：缺少真实值。")
        return

    print(f"{'Method':<15} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'CPC':<8} {'JSD':<8}")
    print("-" * 60)

    special_methods = {"OVS", "OVS-MCM","DeepGravity", "DeepGravity-MCM", "DS", "DS-MCM", "GPT2", "GPT2-MCM", "SEMM", "SEMM wo MCSM"}
    special_methods_new = {"SSM", "RED-SSM", "DUE-GB", "DUE-LS", "SUE-GB", "SUE-LS", "SSM-LS", "SSM-BO"}  # 新的特殊方法组

    for method, pred in predictions_dict.items():
        if method in special_methods and pred.shape[0] == 35 and real.shape[0] == 35:
            # 前32步指标
            rmse1, mae1, mape1, cpc1, jsd1 = evaluate_metrics(pred[:32], real[:32])
            # 后3步指标
            rmse2, mae2, mape2, cpc2, jsd2 = evaluate_metrics(pred[-3:], real[-3:])
            # 平均
            rmse = (rmse1 + rmse2) / 2
            mae = (mae1 + mae2) / 2
            mape = (mape1 + mape2) / 2
            cpc = (cpc1 + cpc2) / 2
            jsd = (jsd1 + jsd2) / 2
            
        elif method in special_methods_new and pred.shape[0] == 35 and real.shape[0] == 35:
            # 对每个样本单独计算指标，然后累加平均
            total_rmse, total_mae, total_mape, total_cpc, total_jsd = 0, 0, 0, 0, 0
            
            for i in range(35):
                sample_pred = pred[i:i+1]  # 获取单个样本预测
                sample_real = real[i:i+1]  # 获取单个样本真实值
                rmse, mae, mape, cpc, jsd = evaluate_metrics(sample_pred, sample_real)
                
                total_rmse += rmse
                total_mae += mae
                total_mape += mape
                total_cpc += cpc
                total_jsd += jsd
            
            # 计算平均值
            rmse = total_rmse / 35
            mae = total_mae / 35
            mape = total_mape / 35
            cpc = total_cpc / 35
            jsd = total_jsd / 35
            
        else:
            if method == "PESL":
                b, n2 = pred.shape
                # print(b, n2)
                assert b == 35 and n2 == 11990  # 缺了110个对角线元

                # 初始化全0矩阵 [b, 110*110]
                pred_full = np.zeros((b, 110 * 110), dtype=pred.dtype)

                # 构建掩码：对角线位置的索引
                full_indices = np.arange(110 * 110).reshape(110, 110)
                diag_indices = np.diag_indices(110)
                diag_flat_indices = full_indices[diag_indices].flatten()  # 长度为110

                # 获取非对角线索引
                all_indices = np.arange(110 * 110)
                non_diag_indices = np.setdiff1d(all_indices, diag_flat_indices)

                # 将预测值填入非对角线位置
                pred_full[:, non_diag_indices] = pred

                # reshape 为 [b, 110, 110]
                pred = pred_full.reshape(b, 110, 110)

                # 对PESL处理后的每个样本单独计算指标，然后累加平均
                total_rmse, total_mae, total_mape, total_cpc, total_jsd = 0, 0, 0, 0, 0
                
                for i in range(35):
                    sample_pred = pred[i:i+1]  # 获取单个样本预测
                    sample_real = real[i:i+1]  # 获取单个样本真实值
                    rmse, mae, mape, cpc, jsd = evaluate_metrics(sample_pred, sample_real)
                    
                    total_rmse += rmse
                    total_mae += mae
                    total_mape += mape
                    total_cpc += cpc
                    total_jsd += jsd
                
                # 计算平均值
                rmse = total_rmse / 35
                mae = total_mae / 35
                mape = total_mape / 35
                cpc = total_cpc / 35
                jsd = total_jsd / 35
                # rmse, mae, mape, cpc, jsd = evaluate_metrics(pred, real)
            else:
                b, n, _= pred.shape
                assert b == 35 and n == 110
                rmse, mae, mape, cpc, jsd = evaluate_metrics(pred, real)

        print(f"{method:<15} {rmse:<8.4f} {mae:<8.4f} {mape:<8.4f} {cpc:<8.4f} {jsd:<8.4f}")

if __name__ == "__main__":
    print("ddd")
    evaluate_all_methods()