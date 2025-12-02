import torch

def nb_zeroinflated_nll_loss(y, n, p, pi, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    https://stats.idre.ucla.edu/r/dae/zinb/
    """
    idx_yeq0 = y == 0
    idx_yg0 = y > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = y[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(n_yg0) + n_yg0 * torch.log(p_yg0) + yg0 * torch.log(1 - p_yg0)

    return -torch.sum(L_yeq0) - torch.sum(L_yg0)


def calculate_rmse_mae(predictions, targets):
    '''

    :param predictions: [T,N,N] T个时间步上的OD矩阵预测值
    :param targets:  [T,N,N] T个时间步上的OD矩阵真值
    :return:
    '''
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
    for t in range(pred_flat.shape[0]):
        pred_t = pred_flat[t]
        targ_t = targ_flat[t]

        epsilon = 1e-8
        pred_dist = (pred_t + epsilon) / (torch.sum(pred_t) + epsilon * pred_t.numel())
        targ_dist = (targ_t + epsilon) / (torch.sum(targ_t) + epsilon * targ_t.numel())

        m = 0.5 * (pred_dist + targ_dist)
        kl1 = torch.sum(pred_dist * torch.log(pred_dist / m))
        kl2 = torch.sum(targ_dist * torch.log(targ_dist / m))
        jsd_t = 0.5 * (kl1 + kl2)
        jsd_list.append(jsd_t.item())
    jsd = sum(jsd_list) / len(jsd_list)


    return rmse.item(), mae.item(), mape.item(),cpc,jsd