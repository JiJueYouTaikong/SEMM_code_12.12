
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np

# 原始数据
layers = [2, 4, 6, 8, 10]
rmse_layer = [7.536, 7.4936, 7.6608, 5.7127, 6.8898]
mae_layer = [0.9963, 0.9305, 0.9516, 0.8276, 0.9042]
cpc_layer = [0.6193, 0.6190, 0.5948, 0.6807, 0.6395]
jsd_layer = [0.1152, 0.1152, 0.1163, 0.1049, 0.1113]

neurons_labels = [64, 128, 256, 512, 1024]
neurons_idx = list(range(len(neurons_labels)))  # 均匀分布的索引
rmse_neuron = [5.9635, 6.0437, 5.7127, 6.2628, 6.0839]
mae_neuron = [0.8542, 0.8881, 0.8276, 0.8922, 0.8688]
cpc_neuron = [0.6655, 0.6690, 0.6807, 0.6608, 0.6762]
jsd_neuron = [0.1068, 0.1084, 0.1049, 0.1090, 0.1062]

# 颜色和标记
# colors = ['#e84d8a', '#64c4eb', '#7f57ae', '#fdb225']
colors = ['#e84d8a', '#64c4eb', '#7f57ae', '#f29f05']

markers = ['o', 's', '^', 'D']
labels = ['RMSE', 'MAE', 'CPC', 'JSD']

fig = plt.figure(figsize=(13, 4), dpi=100)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

def set_fixed_yticks(ax, data, n=6):
    """为ax设置n个均匀分布的刻度，覆盖data范围，且上下边界有刻度"""
    data_min, data_max = min(data), max(data)
    # 适当扩大范围一点点，防止刻度与点重合
    span = data_max - data_min
    if span == 0:
        # 数据全相等时，刻度设置为data±1
        ticks = np.linspace(data_min - 1, data_min + 1, n)
    else:
        # 将扩展比例从0.02增大到0.05，增加上下边界的空白
        lower = data_min - 0.05 * span
        upper = data_max + 0.05 * span
        ticks = np.linspace(lower, upper, n)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    # 格式化显示到小数点后2位
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

def plot_sensitivity(ax, x, x_labels, y_list, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    axes = [ax]
    lines = []

    # 主 y 轴（RMSE）
    y0 = y_list[0]
    set_fixed_yticks(ax, y0, n=6)
    ln0, = ax.plot(x, y0, marker=markers[0], color=colors[0], label=labels[0],
                   markerfacecolor='white', markeredgewidth=2.0, markersize=8, zorder=3)

    lines.append(ln0)
    ax.spines['left'].set_color(colors[0])  # 轴线颜色 ✅
    ax.tick_params(axis='y', colors=colors[0])
    ax.tick_params(axis='y', labelcolor=colors[0])
    ax.yaxis.label.set_color(colors[0])
    ax.spines['left'].set_color(colors[0])



    # 其他3个指标的y轴
    for i in range(1, 4):
        ax_i = ax.twinx()
        ax_i.spines.right.set_position(("axes", 1 + 0.12 * (i - 1)))
        ax_i.set_frame_on(True)
        axes.append(ax_i)

        y = y_list[i]
        # 这里保持自动刻度，也可以调用set_fixed_yticks(ax_i, y, n=6)试试
        ax_i.set_ylim(min(y)*0.95, max(y)*1.05)
        if i ==3:
            ax_i.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        else:
            ax_i.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax_i.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ln, = ax_i.plot(x, y, marker=markers[i], color=colors[i], label=labels[i],
                        markerfacecolor='white', markeredgewidth=2.0, markersize=8, zorder=3)
        lines.append(ln)
        ax_i.tick_params(axis='y', labelcolor=colors[i])
        ax_i.yaxis.label.set_color(colors[i])
        ax_i.spines['right'].set_color(colors[i])
        ax_i.grid(False)

    # 去除所有 y 轴的 label，仅保留图例
    for a in axes:
        a.set_ylabel("")

    return lines  # 返回所有线条以供统一图例

# 子图1
ax0 = fig.add_subplot(gs[0])
lines0 = plot_sensitivity(ax0, layers, layers,
                 [rmse_layer, mae_layer, cpc_layer, jsd_layer],
                 xlabel="(a) Number of Layers")

# 子图2
ax1 = fig.add_subplot(gs[1])
lines1 = plot_sensitivity(ax1, neurons_idx, neurons_labels,
                 [rmse_neuron, mae_neuron, cpc_neuron, jsd_neuron],
                 xlabel="(b) Number of Hidden Units")

# 整图统一图例放在上方中央，横跨两个子图
fig.legend(lines0, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.02), bbox_transform=fig.transFigure)


# 不调用 tight_layout() 避免与 legend 位置冲突
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.15, wspace=0.5)
# plt.savefig("可视化/消融-敏感性/敏感性-折线图.png", bbox_inches="tight", dpi=300)
plt.savefig("可视化/消融-敏感性/敏感性-折线图.pdf", format='pdf',bbox_inches="tight", dpi=300)
plt.show()