import numpy as np

# 加载OD矩阵数据
od = np.load('../data/OD_完整批处理_3.17_Final.npy')  # 形状为[168, 110, 110]


od_filtered = od[-35:-34]  # 现在形状为[133, 110, 110]

departures_per_timestep = np.sum(od_filtered, axis=2)


avg_departures_per_region = np.mean(departures_per_timestep, axis=0)

print(avg_departures_per_region.shape)

# 如果需要保存结果，可以使用以下代码
np.save('../data/初始P估计_N_25.7.6_from1', avg_departures_per_region)