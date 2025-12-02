import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

# -------------------------- 1. 基础配置 & 数据准备 --------------------------
# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11  # 全局基础字号提升
plt.rcParams['axes.grid'] = True  # 全局网格开启
plt.rcParams['grid.alpha'] = 0.5  # 网格透明度
plt.rcParams['grid.linewidth'] = 0.95  # 网格线宽度
# plt.rcParams['grid.color'] = '#E0E0E0'  # 网格线颜色

# 数据定义
models = ['SSM', 'SEMM-SSM']
metrics = ['RMSE', 'MAE', 'CPC', 'JSD', 'Inference Time (s)']
ssm_values = np.array([7.3833, 1.2922, 0.5672, 0.1918, 9358.6992])
semm_ssm_values = np.array([6.866, 1.0325, 0.6461, 0.1117, 0.6343])

# 计算改进率和绝对差值
improvement_rates = []
improvement_abs = []
for i, metric in enumerate(metrics):
    base = ssm_values[i]
    opt = semm_ssm_values[i]
    if metric in ['RMSE', 'MAE', 'JSD', 'Inference Time (s)']:
        rate = (base - opt) / base * 100  # 越低越好
        abs_diff = base - opt
    else:  # CPC: 越高越好
        rate = (opt - base) / base * 100
        abs_diff = opt - base
    improvement_rates.append(round(rate, 4))
    improvement_abs.append(round(abs_diff, 4))

# -------------------------- 2. 绘图配置 --------------------------
# 创建画布 - 更紧凑的尺寸
fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.5), sharey=False)
# fig.subplots_adjust(wspace=0.2, hspace=0.2, top=0.9, bottom=0.15)  # 进一步紧凑布局

# 颜色配置
colors = {'SSM': '#8db7d2', 'SEMM-SSM': '#5e62a9'}
width = 0.22  # 柱形宽度
x_positions = [-0.2, 0.2]  # 调整X轴位置，拉近两个柱形间距（从[0,1]改为[-0.2,0.2]）

# -------------------------- 3. 绘制每个子图 --------------------------
for idx, (ax, metric) in enumerate(zip(axes, metrics)):
    values = [ssm_values[idx], semm_ssm_values[idx]]

    # 绘制柱状图 - 拉近间距
    bars = ax.bar(x_positions, values, width,
                  color=[colors['SSM'], colors['SEMM-SSM']],
                  alpha=1, edgecolor='black', linewidth=1.2)

    # 设置x轴标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, fontsize=12, weight='medium')

    # 设置标题
    ax.set_title(metric, fontsize=14, weight='bold', pad=10)

    # -------------------------- 相对提升值标注 --------------------------
    rate = improvement_rates[idx]
    color = '#a80326' if rate > 0 else '#C62828'
    sign = '+' if rate > 0 else ''

    # SEMM-SSM柱形的位置和高度
    semm_bar = bars[1]
    bar_height = semm_bar.get_height()
    bar_center = semm_bar.get_x() + semm_bar.get_width() / 2

    # 标注位置（靠近柱形顶部）
    if metric == 'Inference Time (s)':
        # 对数刻度的标注位置
        annotation_y = bar_height * 1.2
    else:
        # 线性刻度的标注位置
        annotation_y = bar_height * 1.02

    # 添加改进率标注（仅保留这个）
    ax.text(bar_center, annotation_y,
            f'{sign}{rate:.2f}%',
            ha='center', va='bottom',
            fontsize=13, weight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.9, edgecolor='lightgray', linewidth=0.5))

    # -------------------------- 轴配置 --------------------------
    # Y轴标签
    if idx == 0:  # 只在第一个子图显示Y轴标签
        ax.set_ylabel('Value', fontsize=13, weight='medium')

    # 推理时间使用对数刻度
    if metric == 'Inference Time (s)':
        ax.set_yscale('log')
        min_val = min(values)
        max_val = max(values)
        ax.set_ylim(min_val * 0.1, max_val * 10)

        # 自定义刻度格式
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

        # Y轴刻度标签字号
        ax.tick_params(axis='y', labelsize=11)
    else:
        # 线性刻度配置
        max_value = max(values)
        ax.set_ylim(bottom=0, top=max_value * 1.15)

        # Y轴刻度标签字号
        ax.tick_params(axis='y', labelsize=11)

    # X轴刻度参数
    ax.tick_params(axis='x', labelsize=12)

    # 增强网格显示
    ax.grid(True, which='major', axis='y', alpha=0.4, linewidth=0.8)
    ax.grid(True, which='minor', axis='y', alpha=0.1, linewidth=0.5)
    ax.set_axisbelow(True)  # 网格在柱形下方

    # 设置X轴范围，优化柱形显示
    ax.set_xlim(-0.5, 0.5)

# -------------------------- 图例 --------------------------
ssm_patch = mpatches.Patch(color=colors['SSM'], label='SSM', alpha=0.8)
semm_ssm_patch = mpatches.Patch(color=colors['SEMM-SSM'], label='SEMM-SSM', alpha=0.8)

# fig.legend(handles=[ssm_patch, semm_ssm_patch],
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.02),
#            ncol=2,
#            fontsize=13,
#            frameon=True,
#            fancybox=True,
#            shadow=False,
#            framealpha=0.9,
#            edgecolor='lightgray')

from datetime import datetime
# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
# 紧凑布局
plt.tight_layout()

plt.savefig(
    f'./可视化/结果图/NC-2-scalability-{formatted_time}.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    facecolor='white',  # 背景色为白色（避免透明背景在论文中显示异常）
    edgecolor='none'
)


# plt.show()

# -------------------------- 4. 详细改进率表格 --------------------------
print("=" * 90)
print("Relative Improvement Rate Summary (SEMM-SSM vs SSM)")
print("=" * 90)
print(f"{'Metric':<25} {'Improvement Rate':<20} {'Absolute Difference':<25} {'Interpretation':<20}")
print("-" * 90)

for metric, rate, abs_diff in zip(metrics, improvement_rates, improvement_abs):
    if metric in ['RMSE', 'MAE', 'JSD', 'Inference Time (s)']:
        interpretation = "Lower is better ✓" if rate > 0 else "Higher is worse ✗"
    else:  # CPC
        interpretation = "Higher is better ✓" if rate > 0 else "Lower is worse ✗"

    # 推理时间特殊格式化
    if metric == 'Inference Time (s)':
        rate_str = f"{rate:+.2f}% (Massive improvement)"
        abs_diff_str = f"{abs_diff:,.4f} s ({semm_ssm_values[4] / ssm_values[4]:.2%} of original)"
    else:
        rate_str = f"{rate:+.2f}%"
        abs_diff_str = f"{abs_diff:.4f}"

    print(f"{metric:<25} {rate_str:<20} {abs_diff_str:<25} {interpretation:<20}")