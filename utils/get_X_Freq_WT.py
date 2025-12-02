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

    # *** 参数设置
    K = 50  # 小波系数数量
    J = 3  # 贪婪选择的周期数
    wavelet = 'db4'  # 小波基

    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行小波变换
        coeffs = pywt.wavedec(seq_train[:, i], wavelet)
        flat_coeffs = np.concatenate(coeffs)

        # 计算小波系数的幅值
        wavelet_magnitude = np.abs(flat_coeffs)
        top_k_indices = np.argsort(wavelet_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 贪婪选择最优的 J 个周期
        selected_coeffs = []
        min_val_error = float('inf')  # 初始化最小验证误差为无穷大

        for j in range(J):
            best_coeff_index = None
            for k in top_k_indices:
                if k in selected_coeffs:  # 跳过已选择的系数
                    continue

                temp_coeffs = selected_coeffs + [k]
                new_flat_coeffs = np.zeros_like(flat_coeffs)
                new_flat_coeffs[temp_coeffs] = flat_coeffs[temp_coeffs]

                # 修正这里，生成正确格式的系数结构描述
                coeff_slices = pywt.coeff_slices(coeffs)
                new_coeffs = pywt.array_to_coeffs(new_flat_coeffs, coeff_slices, output_format='wavedec')
                reconstructed_signal = pywt.waverec(new_coeffs, wavelet)[:train_size]

                # 计算验证集误差
                t_val = np.arange(val_size)
                reconstructed_val = reconstructed_signal[:val_size]
                val_error = np.mean((reconstructed_val - seq_val[:, i]) ** 2)

                # 贪婪选择当前最优的系数
                if val_error < min_val_error:
                    min_val_error = val_error
                    best_coeff_index = k

            if best_coeff_index is not None:
                selected_coeffs.append(best_coeff_index)

        # 存储贪婪选择的系数
        selected_periods.append(selected_coeffs)

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的系数
        selected_coeffs = selected_periods[i]
        flat_coeffs = np.concatenate(pywt.wavedec(seq_train[:, i], wavelet))
        new_flat_coeffs = np.zeros_like(flat_coeffs)
        new_flat_coeffs[selected_coeffs] = flat_coeffs[selected_coeffs]

        # 修正这里，生成正确格式的系数结构描述
        coeff_slices = pywt.coeff_slices(pywt.wavedec(seq_train[:, i], wavelet))
        new_coeffs = pywt.array_to_coeffs(new_flat_coeffs, coeff_slices, output_format='wavedec')
        reconstructed_train = pywt.waverec(new_coeffs, wavelet)[:train_size]

        z_list.append(reconstructed_train)

    z_t = np.array(z_list)
    z_t = z_t.T
    z_t_mean = z_t.mean(axis=0)

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

    # *** 参数设置
    K = 50  # 小波系数数量
    J = 3  # 贪婪选择的周期数
    wavelet = 'db4'  # 小波基

    selected_periods = []  # 存储每个区域贪婪选择的周期信息

    for i in range(N):
        # 对每个区域进行小波变换
        coeffs = pywt.wavedec(seq_train[:, i], wavelet)
        flat_coeffs = np.concatenate(coeffs)

        # 计算小波系数的幅值
        wavelet_magnitude = np.abs(flat_coeffs)
        top_k_indices = np.argsort(wavelet_magnitude)[::-1][:K]  # 幅值从大到小排序，取前 K 个

        # 贪婪选择最优的 J 个周期
        selected_coeffs = []
        min_val_error = float('inf')  # 初始化最小验证误差为无穷大

        for j in range(J):
            best_coeff_index = None
            for k in top_k_indices:
                if k in selected_coeffs:  # 跳过已选择的系数
                    continue

                temp_coeffs = selected_coeffs + [k]
                new_flat_coeffs = np.zeros_like(flat_coeffs)
                new_flat_coeffs[temp_coeffs] = flat_coeffs[temp_coeffs]

                # 修正这里，生成正确格式的系数结构描述
                coeff_slices = pywt.coeff_slices(coeffs)
                new_coeffs = pywt.array_to_coeffs(new_flat_coeffs, coeff_slices, output_format='wavedec')
                reconstructed_signal = pywt.waverec(new_coeffs, wavelet)[:train_size]

                # 计算验证集误差
                t_val = np.arange(val_size)
                reconstructed_val = reconstructed_signal[:val_size]
                val_error = np.mean((reconstructed_val - seq_val[:, i]) ** 2)

                # 贪婪选择当前最优的系数
                if val_error < min_val_error:
                    min_val_error = val_error
                    best_coeff_index = k

            if best_coeff_index is not None:
                selected_coeffs.append(best_coeff_index)

        # 存储贪婪选择的系数
        selected_periods.append(selected_coeffs)

    z_list = []
    # *** 2. 信号重建与可视化
    for i in range(N):
        # 提取贪婪选择的系数
        selected_coeffs = selected_periods[i]
        flat_coeffs = np.concatenate(pywt.wavedec(seq_train[:, i], wavelet))
        new_flat_coeffs = np.zeros_like(flat_coeffs)
        new_flat_coeffs[selected_coeffs] = flat_coeffs[selected_coeffs]

        # 修正这里，生成正确格式的系数结构描述
        coeff_slices = pywt.coeff_slices(pywt.wavedec(seq_train[:, i], wavelet))
        new_coeffs = pywt.array_to_coeffs(new_flat_coeffs, coeff_slices, output_format='wavedec')
        reconstructed_train = pywt.waverec(new_coeffs, wavelet)[:train_size]

        z_list.append(reconstructed_train)

    z_t = np.array(z_list)
    z_t = z_t.T
    z_t_mean = z_t.mean(axis=0)

    return z_t_mean
