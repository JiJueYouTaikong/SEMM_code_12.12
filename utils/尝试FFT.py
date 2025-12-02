import numpy as np
import matplotlib.pyplot as plt

# 生成7天每小时采样一次的数据
num_samples = 7 * 24
t = np.arange(num_samples)
# 生成一个简单的包含一些频率成分的速度序列数据
speed = 5 + 2 * np.sin(2 * np.pi * 1 / 24 * t) + 1 * np.sin(2 * np.pi * 1 / 12 * t) + np.random.normal(0, 0.5,
                                                                                                       num_samples)

# 进行FFT变换
fft_result = np.fft.fft(speed)
frequencies = np.fft.fftfreq(num_samples)

# 取正频率部分
positive_freqs = frequencies[:num_samples // 2]
positive_fft = 2 / num_samples * np.abs(fft_result[:num_samples // 2])

# 找出前3个显著的峰值
num_prominent = 3
indices = np.argsort(positive_fft)[-num_prominent:][::-1]
prominent_freqs = positive_freqs[indices]
prominent_amplitudes = positive_fft[indices]
prominent_periods = 1 / prominent_freqs

# 输出前几个显著的周期值、频率值、振幅
print("前几个显著的周期值 (小时):", prominent_periods)
print("前几个显著的频率值 (cycles per hour):", prominent_freqs)
print("前几个显著的振幅:", prominent_amplitudes)

# 绘制原始数据
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, speed)
plt.title('Original Speed Data')
plt.xlabel('Time (hours)')
plt.ylabel('Speed')

# 绘制FFT结果
plt.subplot(2, 1, 2)
# 修改此处，使用 plt.scatter 绘制点
plt.scatter(positive_freqs, positive_fft)
plt.title('FFT Magnitude Spectrum')
plt.xlabel('Frequency (cycles per hour)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

print(positive_freqs)
print(positive_fft)