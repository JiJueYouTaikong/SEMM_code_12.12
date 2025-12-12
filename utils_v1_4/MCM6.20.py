import numpy as np

# 加载数据
od = np.load('../data/OD_完整批处理_3.17_Final.npy')  # 形状 [T, N, N]

od = od[:100]

# 在时间维度 (axis=0) 求均值，得到 [N, N]
od_mean = np.mean(od, axis=0)

# 将均值矩阵展平为一维
flat_mean = od_mean.flatten()

# 找出均值最大的 100 个 OD pair （未保证顺序）
topk_indices_flat = np.argpartition(-flat_mean, 100)[:100]

# 按均值从大到小排序
topk_indices_flat = topk_indices_flat[np.argsort(-flat_mean[topk_indices_flat])]

# 将展平索引转回 (i, j)
topk_indices = np.array(np.unravel_index(topk_indices_flat, od_mean.shape)).T

# 输出
print("【限定前100时间步】均值最大的 100 个 OD pair 及其均值：")
for idx, (i, j) in enumerate(topk_indices):
    value = od_mean[i, j]
    print(f"{idx+1:3d}: (i={i}, j={j}), mean={value:.4f}")

# # 将三维数组展平
# flat_od = od.flatten()
#
#
# # 获取最大的 20 个值在展开后数组中的下标
# topk_indices_flat = np.argpartition(-flat_od, 200)[:200]
#
# # 对这 20 个值按实际值降序排序（可选，保证顺序正确）
# topk_indices_flat = topk_indices_flat[np.argsort(-flat_od[topk_indices_flat])]
#
# # 将一维下标转换回三维下标 (i, j, k)
# topk_indices = np.array(np.unravel_index(topk_indices_flat, od.shape)).T
#
# # 输出结果
# print("最大的 30 个值及其对应下标 (i, j, k)：")
# for idx, (i, j, k) in enumerate(topk_indices):
#     value = od[i, j, k]
#     print(f"{idx+1:2d}: (i={i}, j={j}, k={k}), value={value}")
# 训练集20-28   67-68
# 验证集66-58  95-94  94-93  20-28  67-68
# 测试集41-56  20-28   67-68