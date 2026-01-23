import numpy as np

# 1. 加载三维数据 (t, n, n)
tnn_data = np.load("../dataset/Manhattan_Flow_TNN_15min_0501_0731_MSA-SUE_logit3_最终FINAL.npy")

# 查看数据形状，确认维度信息
print(f"原始数据形状: {tnn_data.shape}")
t, n, _ = tnn_data.shape  # 解包时间步t、行列数n


# 2. 核心逻辑：对每个n×n矩阵，计算每个i对应的「第i行+第i列（去重交叉点）的非零平均值」
def row_col_combined_nonzero_mean(matrix):
    """
    计算二维矩阵中，每个i对应的第i行+第i列（交叉点去重）的非零元素平均值
    :param matrix: 形状为 (n, n) 的二维数组
    :return: 形状为 (n,) 的一维数组，每个i对应行+列的非零平均值
    """
    n = matrix.shape[0]
    result = np.zeros(n, dtype=np.float64)  # 初始化结果数组

    for i in range(n):
        # 步骤1：提取第i行和第i列
        row_i = matrix[i, :]  # 第i行，形状 (n,)
        col_i = matrix[:, i]  # 第i列，形状 (n,)

        # 步骤2：合并行和列，去除交叉点 (i,i) 的重复值（只保留1次）
        # 方法：行 + 列中排除索引i的元素（即去掉交叉点重复项）
        combined = np.concatenate([row_i, col_i[:i], col_i[i + 1:]])

        # 步骤3：计算合并后数组的非零和与非零个数
        nonzero_combined = combined[combined != 0]  # 筛选非零元素
        nonzero_sum = nonzero_combined.sum()
        nonzero_count = len(nonzero_combined)

        # 步骤4：安全计算平均值，避免除以0 【将平均改为了 求和】
        if nonzero_count > 0:
            # result[i] = nonzero_sum / nonzero_count
            result[i] = nonzero_sum  # 【将平均改为了 求和】

        else:
            result[i] = 0.0  # 无非零元素时赋值0

    return result


# 3. 遍历所有时间步，应用上述函数，得到 (t, n) 矩阵
tn_matrix = np.zeros((t, n), dtype=np.float64)  # 初始化结果数组
for time_step in range(t):
    current_nn_matrix = tnn_data[time_step]  # 获取当前时间步的 n×n 矩阵
    tn_matrix[time_step] = row_col_combined_nonzero_mean(current_nn_matrix)  # 计算并保存结果

# 4. 保存结果
save_path = "../dataset/Manhattan_Flow_TN_聚合后的最终输入_15min_0501_0731_MSA-SUE_logit3_最终FINAL.npy"
np.save(save_path, tn_matrix)

print(tn_matrix[0,:20])


print(tn_matrix[2,:20])

print(f"处理后数据形状: {tn_matrix.shape}")
print(f"数据已保存至: {save_path}")

# -------------- 验证你的示例（完全匹配你的预期结果） --------------
test_nn = np.array([[0, 1, 2], [4, 0, 0], [0, 0, 0]])
test_avg = row_col_combined_nonzero_mean(test_nn)
print(f"\n示例矩阵的结果: {test_avg}")