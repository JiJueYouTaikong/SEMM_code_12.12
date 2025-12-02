import numpy as np
import matplotlib.pyplot as plt
import pywt

def get_X_Freq(seq):
    # *** 数据分割
    T, N = seq.shape
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    seq_train, seq_val, seq_test = seq[train_indices], seq[val_indices], seq[test_indices]

    #print("Training set: seq shape", seq_train.shape)
    #print("Validation set: seq shape", seq_val.shape)
    #print("Test set: seq shape", seq_test.shape)

    # *** 参数设置
    K = 50  # 小波系数基数
    J = 3  # 贪婪选择的周期数

    # *** 小波变换 (Wavelet Transform)
    wavelet_coefficients = np.zeros((K, N))  # 保存 top K 的小波系数
    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行小波变换
        coeffs = pywt.wavedec(seq_train[:, i], 'db4', level=5)
        flat_coeffs = np.concatenate(coeffs)

        # 计算小波系数的幅值
        wavelet_magnitude = np.abs(flat_coeffs)
        top_k_indices = np.argsort(wavelet_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 保存 top K 系数
        wavelet_coefficients[:K, i] = flat_coeffs[top_k_indices]

        # 贪婪选择最优的 J 个周期
        selected_freqs = []
        selected_amps = []
        selected_phases = []

        for j in range(J):
            best_freq = None
            best_amp = None
            best_phase = None
            min_val_error = float('inf')  # 初始化最小验证误差为无穷大

            for k in top_k_indices:
                if k in selected_freqs:  # 跳过已选择的频率
                    continue

                # 当前频率对应的小波系数
                coeff = flat_coeffs[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k

                # 重建信号
                t_train = np.arange(train_size)
                reconstructed_signal = np.zeros_like(t_train, dtype=np.float64)

                # 加上已选频率和当前频率对应的信号分量
                for f, a, p in zip(selected_freqs + [freq], selected_amps + [amp], selected_phases + [phase]):
                    reconstructed_signal += a * np.cos(2 * np.pi * f * t_train + p)

                # 加上直流分量 A0
                A0 = flat_coeffs[0]
                reconstructed_signal += A0

                # 计算验证集误差
                t_val = np.arange(val_size)
                reconstructed_val = reconstructed_signal[:val_size]
                val_error = np.mean((reconstructed_val - seq_val[:, i]) ** 2)

                # 贪婪选择当前最优的频率
                if val_error < min_val_error:
                    min_val_error = val_error
                    best_freq = freq
                    best_amp = amp
                    best_phase = phase

            # 保存当前最优选择
            selected_freqs.append(best_freq)
            selected_amps.append(best_amp)
            selected_phases.append(best_phase)

        # 存储贪婪选择的频率、幅值和相位
        selected_periods.append((selected_freqs, selected_amps, selected_phases))

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的频率、幅值和相位
        selected_freqs, selected_amps, selected_phases = selected_periods[i]

        # 重建训练集信号
        t_train = np.arange(train_size)
        reconstructed_train = np.zeros_like(t_train, dtype=np.float64)

        # 加上直流分量 A0
        coeffs = pywt.wavedec(seq_train[:, i], 'db4', level=5)
        flat_coeffs = np.concatenate(coeffs)
        A0 = flat_coeffs[0]
        reconstructed_train += A0

        for j in range(J):
            freq = selected_freqs[j]
            amp = selected_amps[j]
            phase = selected_phases[j]
            reconstructed_train += amp * np.cos(2 * np.pi * freq * t_train + phase)

        z_list.append(reconstructed_train)

        # if i % 1000 == 0:
        #     # 可视化训练集原始信号与重建信号
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(seq_train[:200, i], label='Original Signal')
        #     plt.plot(reconstructed_train[:200], label='Reconstructed Signal', linestyle='--')
        #     plt.title(f"Region {i + 1} - Training Set Reconstruction")
        #     plt.xlabel("Time Step")
        #     plt.ylabel("OD")
        #     plt.legend()
        #     plt.show()

    z_t = np.array(z_list)
    z_t = z_t.T
    z_t_mean = z_t.mean(axis=0)
    #print(z_t_mean.shape)

    return z_t_mean



def get_X_Freq_MCM(seq):
    # *** 数据分割
    T, N = seq.shape
    train_size = int(T * 0.9537)
    val_size = int(T * 0.0225)

    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    seq_train, seq_val, seq_test = seq[train_indices], seq[val_indices], seq[test_indices]

    #print("Training set: seq shape", seq_train.shape)
    #print("Validation set: seq shape", seq_val.shape)
    #print("Test set: seq shape", seq_test.shape)

    # *** 参数设置
    K = 50  # 小波系数基数
    J = 3  # 贪婪选择的周期数

    # *** 小波变换 (Wavelet Transform)
    wavelet_coefficients = np.zeros((K, N))  # 保存 top K 的小波系数
    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行小波变换
        coeffs = pywt.wavedec(seq_train[:, i], 'db4', level=5)
        flat_coeffs = np.concatenate(coeffs)

        # 计算小波系数的幅值
        wavelet_magnitude = np.abs(flat_coeffs)
        top_k_indices = np.argsort(wavelet_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 保存 top K 系数
        wavelet_coefficients[:K, i] = flat_coeffs[top_k_indices]

        # 贪婪选择最优的 J 个周期
        selected_freqs = []
        selected_amps = []
        selected_phases = []

        for j in range(J):
            best_freq = None
            best_amp = None
            best_phase = None
            min_val_error = float('inf')  # 初始化最小验证误差为无穷大

            for k in top_k_indices:
                if k in selected_freqs:  # 跳过已选择的频率
                    continue

                # 当前频率对应的小波系数
                coeff = flat_coeffs[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k

                # 重建信号
                t_train = np.arange(train_size)
                reconstructed_signal = np.zeros_like(t_train, dtype=np.float64)

                # 加上已选频率和当前频率对应的信号分量
                for f, a, p in zip(selected_freqs + [freq], selected_amps + [amp], selected_phases + [phase]):
                    reconstructed_signal += a * np.cos(2 * np.pi * f * t_train + p)

                # 加上直流分量 A0
                A0 = flat_coeffs[0]
                reconstructed_signal += A0

                # 计算验证集误差
                t_val = np.arange(val_size)
                reconstructed_val = reconstructed_signal[:val_size]
                val_error = np.mean((reconstructed_val - seq_val[:, i]) ** 2)

                # 贪婪选择当前最优的频率
                if val_error < min_val_error:
                    min_val_error = val_error
                    best_freq = freq
                    best_amp = amp
                    best_phase = phase

            # 保存当前最优选择
            selected_freqs.append(best_freq)
            selected_amps.append(best_amp)
            selected_phases.append(best_phase)

        # 存储贪婪选择的频率、幅值和相位
        selected_periods.append((selected_freqs, selected_amps, selected_phases))

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的频率、幅值和相位
        selected_freqs, selected_amps, selected_phases = selected_periods[i]

        # 重建训练集信号
        t_train = np.arange(train_size)
        reconstructed_train = np.zeros_like(t_train, dtype=np.float64)

        # 加上直流分量 A0
        coeffs = pywt.wavedec(seq_train[:, i], 'db4', level=5)
        flat_coeffs = np.concatenate(coeffs)
        A0 = flat_coeffs[0]
        reconstructed_train += A0

        for j in range(J):
            freq = selected_freqs[j]
            amp = selected_amps[j]
            phase = selected_phases[j]
            reconstructed_train += amp * np.cos(2 * np.pi * freq * t_train + phase)

        z_list.append(reconstructed_train)

        # if i % 1000 == 0:
        #     # 可视化训练集原始信号与重建信号
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(seq_train[:200, i], label='Original Signal')
        #     plt.plot(reconstructed_train[:200], label='Reconstructed Signal', linestyle='--')
        #     plt.title(f"Region {i + 1} - Training Set Reconstruction")
        #     plt.xlabel("Time Step")
        #     plt.ylabel("OD")
        #     plt.legend()
        #     plt.show()

    z_t = np.array(z_list)
    z_t = z_t.T
    z_t_mean = z_t.mean(axis=0)
    #print(z_t_mean.shape)

    return z_t_mean

