# import numpy as np
# from scipy.signal import stft
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 生成时间序列
# T = 168
# t = np.linspace(0, 1, T, endpoint=False)
#
# # 多频率成分信号
# x = np.sin(2 * np.pi * 5 * t)
# x += 0.5 * np.sin(2 * np.pi * 10 * t)
# x += 0.3 * np.sin(2 * np.pi * 15 * t)
# x += 0.2 * np.sin(2 * np.pi * 20 * t)
# x += 0.4 * np.sin(2 * np.pi * (5 + 10 * t) * t)  # 扫频信号
#
# # 添加瞬态信号
# burst = np.zeros_like(t)
# burst[50:70] = 0.8 * np.sin(2 * np.pi * 30 * t[50:70])
# burst[120:140] = 0.6 * np.sin(2 * np.pi * 25 * t[120:140])
# x += burst
#
# # STFT 计算
# f, t_stft, Zxx = stft(x, fs=1)
# magnitude = np.abs(Zxx)
#
# # 创建3D图形
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 准备颜色映射 - 基于频率
# norm = plt.Normalize(vmin=f.min(), vmax=f.max())
# cmap = plt.cm.plasma
#
# # 对每个时间点绘制频率-幅度曲线
# for i, time in enumerate(t_stft):
#     # 获取当前时间点的所有频率和幅度
#     freqs = f
#     mags = magnitude[:, i]
#
#     # 绘制该时间点的频率-幅度曲线（平滑折线）
#     ax.plot([time] * len(freqs), freqs, mags,
#             color='gray', alpha=0.3, linewidth=0.5)  # 用浅灰色连接
#
#     # 为每个频率点添加颜色标记（基于频率值）
#     colors = cmap(norm(freqs))
#     for freq, mag, color in zip(freqs, mags, colors):
#         ax.plot([time, time], [freq, freq], [0, mag],
#                 color=color, linewidth=1.5)
#
# # 美化图形
# ax.set_title('STFT: Frequency vs Time with Magnitude (Color by Frequency)', fontsize=14)
# ax.set_xlabel('Time [s]', fontsize=12)
# ax.set_ylabel('Frequency [Hz]', fontsize=12)
# ax.set_zlabel('Magnitude', fontsize=12)
#
# # 添加颜色条（表示频率）
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('Frequency [Hz]', fontsize=12)
#
# # 调整视角
# ax.view_init(elev=30, azim=45)
# plt.tight_layout()
# plt.show()

import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 数据加载函数
def load_data(data_name):
    try:
        file_path = f"../可视化/频域/{data_name}.npy"
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            print(f"错误: 文件 {file_path} 不存在")
            return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


# 加载数据
f1 = load_data("f1")
f2 = load_data("f2")
t_stft1 = load_data("t_stft1")
t_stft2 = load_data("t_stft2")
amplitude1 = load_data("amplitude1")
amplitude2 = load_data("amplitude2")

# 检查数据是否完整加载
if any(data is None for data in [f1, f2, t_stft1, t_stft2, amplitude1, amplitude2]):
    print("数据加载不完整，无法继续绘制图形")
else:
    # 定义绘图函数
    def plot_3d_frequency(f, t_stft, amplitude, title, fig_num):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 获取频率范围用于颜色映射
        f_min, f_max = f.min(), f.max()

        # 为每个时间点绘制平滑的频率曲线
        for i in range(amplitude.shape[1]):
            # 获取当前时间点的频率数据
            freq_data = amplitude[:, i]

            # 使用三次样条插值平滑曲线
            from scipy.interpolate import make_interp_spline
            xnew = np.linspace(f.min(), f.max(), 300)  # 插值后的点数
            spl = make_interp_spline(f, freq_data, k=3)  # 三次样条
            freq_smooth = spl(xnew)

            # 根据频率值确定颜色
            for j in range(len(xnew) - 1):
                # 当前线段的平均频率作为颜色依据
                avg_freq = (xnew[j] + xnew[j + 1]) / 2
                color_value = (avg_freq - f_min) / (f_max - f_min)
                color = cm.jet(color_value)

                # 绘制线段
                ax.plot([xnew[j], xnew[j + 1]],
                        [t_stft[i], t_stft[i]],
                        [freq_smooth[j], freq_smooth[j + 1]],
                        color=color, linewidth=1.5)

        # 设置坐标轴标签
        ax.set_xlabel('频率', fontsize=12)
        ax.set_ylabel('时间', fontsize=12)
        ax.set_zlabel('振幅', fontsize=12)
        ax.set_title(title, fontsize=15)

        # 添加颜色条表示频率大小
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(f)
        cbar = plt.colorbar(m, ax=ax, shrink=0.7, aspect=10)
        cbar.set_label('频率大小', rotation=270, labelpad=20)

        # # 隐藏背景网格
        # ax.grid(False)
        # # 移除灰色背景
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False

        # 设置视角
        ax.view_init(elev=30, azim=-80)

        return fig


    # 绘制第一张图
    fig1 = plot_3d_frequency(f1, t_stft1, amplitude1, "速度数据时频振幅", 1)

    # 绘制第二张图
    fig2 = plot_3d_frequency(f2, t_stft2, amplitude2, "总出发量数据时频振幅", 2)

    plt.tight_layout()
    plt.show()