import numpy as np

# # 定义矩阵的行数 N
# N = 110  # 你可以根据需要修改 N 的值
#
# # 生成一个形状为 [N,] 且元素范围在 0 - 90 之间的随机矩阵
# realtime_speed = np.random.randint(0, 80, N)
#
#
#
# # 保存矩阵为 npy 文件
# np.save('realtime_speed.npy', realtime_speed)
#
# speed = np.load('Speed_完整批处理_3.17_Final.npy')
#
# realtime_speed = speed[-2,:]
# print(realtime_speed)
# print(realtime_speed.shape)
#
# np.save('realtime_speed_25.3.18.npy', realtime_speed)
# print(f"矩阵已保存为 realtime_speed.npy，形状为 {realtime_speed.shape}")

od = np.load('OD_完整批处理_3.17_Final.npy')

realtime_od = od[-2]
print(realtime_od)
print(realtime_od.shape)

np.save('初始OD估计_NN_25.3.18.npy', realtime_od)
print(f"矩阵已保存为 realtime_speed.npy，形状为 {realtime_od.shape}")