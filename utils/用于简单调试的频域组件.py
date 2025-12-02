import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


def get_X_Freq(seq, selection_type=1,is_mcm=False):
    # *** 数据分割
    T, N = seq.shape
    if is_mcm:
        train_size = int(T * 0.9537)
        val_size = int(T * 0.0225)
    else:
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
    K = 50  # DCT基数
    J = 3  # 贪婪选择的周期数

    # *** 离散余弦变换 (DCT)
    dct_coefficients = np.zeros((K, N))  # 保存 top K 的 DCT 系数
    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行 DCT
        dct_result = dct(seq_train[:, i], type=2, norm='ortho')

        # 计算 DCT 系数的幅值
        dct_magnitude = np.abs(dct_result)
        top_k_indices = np.argsort(dct_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 计算初始 top-k 周期值
        top_k_periods = [T / k if k != 0 else float('inf') for k in top_k_indices]  # k=0 时周期为无穷大

        # print(f"Region {i + 1}: Initial Top-{K} Periods (before greedy selection): {top_k_periods[:J]}")

        # 保存 top K 系数
        dct_coefficients[:K, i] = dct_result[top_k_indices]

        # 选择最优的 J 个周期
        selected_freqs = []
        selected_amps = []
        selected_phases = []

        if selection_type == 1:
            # 贪婪选择最优的 J 个周期
            for j in range(J):
                best_freq = None
                best_amp = None
                best_phase = None
                min_val_error = float('inf')  # 初始化最小验证误差为无穷大

                for k in top_k_indices:
                    if k in selected_freqs:  # 跳过已选择的频率
                        continue

                    # 当前频率对应的 DCT 系数
                    coeff = dct_result[k]
                    amp = np.abs(coeff)
                    phase = np.angle(coeff)
                    freq = k / T  # 归一化频率

                    # 重建信号
                    t_train = np.arange(train_size)
                    reconstructed_signal = np.zeros_like(t_train, dtype=np.float64)

                    # 加上已选频率和当前频率对应的信号分量
                    for f, a, p in zip(selected_freqs + [freq], selected_amps + [amp], selected_phases + [phase]):
                        reconstructed_signal += a * np.cos(2 * np.pi * f * t_train + p)

                    # 加上直流分量 A₀
                    A0 = dct_result[0] / np.sqrt(train_size)  # DCT 系数的直流分量
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
        elif selection_type == 2:
            # 直接保留Topk中最大的TopJ个
            top_j_indices = top_k_indices[:J]
            for k in top_j_indices:
                coeff = dct_result[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k / T  # 归一化频率
                selected_freqs.append(freq)
                selected_amps.append(amp)
                selected_phases.append(phase)
        elif selection_type == 3:
            # 直接在Topk中随机选择J个
            selected_k_indices = np.random.choice(top_k_indices, J, replace=False)
            for k in selected_k_indices:
                coeff = dct_result[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k / T  # 归一化频率
                selected_freqs.append(freq)
                selected_amps.append(amp)
                selected_phases.append(phase)

        # 存储贪婪选择的频率、幅值和相位
        selected_periods.append((selected_freqs, selected_amps, selected_phases))

        # 计算贪婪选择后的周期值
        # greedy_selected_periods = [T / (freq * T) for freq in selected_freqs]
        greedy_selected_periods = []
        for freq in selected_freqs:
            if freq == 0:
                greedy_selected_periods.append(float('inf'))
            else:
                greedy_selected_periods.append(T / (freq * T))
        # print(f"Region {i + 1}: Greedy Selected Periods (after selection): {greedy_selected_periods}")

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的频率、幅值和相位
        selected_freqs, selected_amps, selected_phases = selected_periods[i]

        # 重建训练集信号
        t_train = np.arange(train_size)
        reconstructed_train = np.zeros_like(t_train, dtype=np.float64)

        # 加上直流分量 A₀
        A0 = dct(seq_train[:, i], type=2, norm='ortho')[0] / np.sqrt(train_size)
        reconstructed_train += A0
        # print(f"区域{i}的A0：{A0}")

        for j in range(J):
            freq = selected_freqs[j]
            amp = selected_amps[j]
            phase = selected_phases[j]
            reconstructed_train += amp * np.cos(2 * np.pi * freq * t_train + phase)
            # print(f"第{j}个周期贡献值：{(amp * np.cos(2 * np.pi * freq * t_train + phase))[:3].astype(int)}")
        # print("最终周期状态")
        z_list.append(reconstructed_train)

        if i % 20 == 0:
            # 可视化训练集原始信号与重建信号
            plt.figure(figsize=(10, 6))
            plt.plot(seq_train[:200, i], label='Original Signal')
            plt.plot(reconstructed_train[:200], label='Reconstructed Signal', linestyle='--')
            plt.title(f"Region {i} - Training Set Reconstruction")
            plt.xlabel("Time Step")
            plt.ylabel("OD")
            plt.legend()
            plt.show()

    # print(len(z_list))
    z_t = np.array(z_list)
    # print(z_t.shape)
    z_t = z_t.T
    # print(z_t.shape)
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
    K = 50  # DCT基数
    J = 3  # 贪婪选择的周期数

    # *** 离散余弦变换 (DCT)
    dct_coefficients = np.zeros((K, N))  # 保存 top K 的 DCT 系数
    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行 DCT
        dct_result = dct(seq_train[:, i], type=2, norm='ortho')

        # 计算 DCT 系数的幅值
        dct_magnitude = np.abs(dct_result)
        top_k_indices = np.argsort(dct_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 计算初始 top-k 周期值
        top_k_periods = [T / k if k != 0 else float('inf') for k in top_k_indices]  # k=0 时周期为无穷大

        # print(f"Region {i + 1}: Initial Top-{K} Periods (before greedy selection): {top_k_periods[:J]}")

        # 保存 top K 系数
        dct_coefficients[:K, i] = dct_result[top_k_indices]

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

                # 当前频率对应的 DCT 系数
                coeff = dct_result[k]
                amp = np.abs(coeff)
                phase = np.angle(coeff)
                freq = k / T  # 归一化频率

                # 重建信号
                t_train = np.arange(train_size)
                reconstructed_signal = np.zeros_like(t_train, dtype=np.float64)

                # 加上已选频率和当前频率对应的信号分量
                for f, a, p in zip(selected_freqs + [freq], selected_amps + [amp], selected_phases + [phase]):
                    reconstructed_signal += a * np.cos(2 * np.pi * f * t_train + p)

                # 加上直流分量 A₀
                A0 = dct_result[0] / np.sqrt(train_size)  # DCT 系数的直流分量
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

        # 计算贪婪选择后的周期值
        # greedy_selected_periods = [T / (freq * T) for freq in selected_freqs]
        greedy_selected_periods = []
        for freq in selected_freqs:
            if freq == 0:
                greedy_selected_periods.append(float('inf'))
            else:
                greedy_selected_periods.append(T / (freq * T))
        # print(f"Region {i + 1}: Greedy Selected Periods (after selection): {greedy_selected_periods}")

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的频率、幅值和相位
        selected_freqs, selected_amps, selected_phases = selected_periods[i]

        # 重建训练集信号
        t_train = np.arange(train_size)
        reconstructed_train = np.zeros_like(t_train, dtype=np.float64)

        # 加上直流分量 A₀
        A0 = dct(seq_train[:, i], type=2, norm='ortho')[0] / np.sqrt(train_size)
        reconstructed_train += A0
        # print(f"区域{i}的A0：{A0}")

        for j in range(J):
            freq = selected_freqs[j]
            amp = selected_amps[j]
            phase = selected_phases[j]
            reconstructed_train += amp * np.cos(2 * np.pi * freq * t_train + phase)
            # print(f"第{j}个周期贡献值：{(amp * np.cos(2 * np.pi * freq * t_train + phase))[:3].astype(int)}")
        # print("最终周期状态")
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

    # print(len(z_list))
    z_t = np.array(z_list)
    # print(z_t.shape)
    z_t = z_t.T
    # print(z_t.shape)
    z_t_mean = z_t.mean(axis=0)
    #print(z_t_mean.shape)

    return z_t_mean

speed = np.load('../data/Speed_完整批处理_3.17_Final.npy')
od = np.load('../data/OD_完整批处理_3.17_Final.npy')

od_production = np.sum(od, axis=-1)  # 形状 [T_train, N]

od_freq = get_X_Freq(od_production,3,False)