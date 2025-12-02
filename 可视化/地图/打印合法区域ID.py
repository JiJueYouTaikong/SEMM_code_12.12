import numpy as np

speed = np.load("武汉速度数据集_1KM_110区域_25.1.14.npy")
print(speed[0,:,1])
print(speed[0,:,1].shape)

np.save("网格映射v1.npy",speed[0,:,1])
# od = np.load("WH_index.npy")
# print(od)
# print(od.shape)