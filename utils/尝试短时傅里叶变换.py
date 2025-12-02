import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

# 生成一个示例时间序列
T = 168
t = np.linspace(0, 1, T, endpoint=False)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# 进行短时傅里叶变换
f, t_stft, Zxx = stft(x, fs=1)

magnitude = np.abs(Zxx)
phase = np.angle(Zxx)


print(f.shape)  # 频率数组 (n_frequencies,)  n_frequencies = nperseg // 2 + 1
print(t_stft.shape)  # 时间数组 (n_times,)  (T - noverlap) // (nperseg - noverlap)
print(Zxx.shape)  # 幅度和相位 (n_frequencies, n_times)





plt.figure(figsize=(10, 4))
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude Spectrum')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [h]')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pywt  # 导入PyWavelets库

# 生成一个示例时间序列
T = 168
t = np.linspace(0, 1, T, endpoint=False)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# 进行小波变换
wavelet = 'morl'  # 选择Morlet小波，适合频率分析
scales = np.arange(1, 64)  # 定义尺度范围，影响频率分辨率
coef, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1)

magnitude = np.abs(coef)  # 小波系数的幅度
phase = np.angle(coef)    # 小波系数的相位

print(freqs.shape)  # 频率数组 (n_scales,)
print(t.shape)      # 时间数组 (T,)
print(coef.shape)   # 小波系数 (n_scales, T)

# 绘制小波变换结果
plt.figure(figsize=(10, 4))
plt.pcolormesh(t, freqs, magnitude, shading='gouraud')
plt.title('Wavelet Transform Magnitude Spectrum')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [h]')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()