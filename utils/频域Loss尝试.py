import numpy as np
import matplotlib.pyplot as plt
import torch

# 生成7天每小时采样一次的数据
num_samples = 7 * 24
num_samples = 12
t = np.arange(num_samples)
# 生成一个简单的包含一些频率成分的速度序列数据
speed = 5 + 2 * np.sin(2 * np.pi * 1 / 24 * t) + 1 * np.sin(2 * np.pi * 1 / 12 * t) + np.random.normal(0, 0.5,num_samples)

print(speed.shape)
speed = torch.Tensor(speed)

fft_speed = torch.fft.rfft(speed)
print(fft_speed.shape)
print(fft_speed)