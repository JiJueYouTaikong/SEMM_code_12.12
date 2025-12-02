import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# 加载数据
od = np.load('./data/OD_完整批处理_3.17_Final.npy')  # 形状 [T, N, N]，T 为 168（24×7）

# 时间区间划分（周中 7-9、17-19 为高峰，其余为平峰）
all_time = np.arange(168)
peak_time_indices = []
for day in range(5):  # 周一到周五（索引 0-4）
    peak_morning = [day * 24 + 7, day * 24 + 8]   # 7-9 点
    peak_evening = [day * 24 + 17, day * 24 + 18] # 17-19 点
    peak_time_indices.extend(peak_morning)
    peak_time_indices.extend(peak_evening)
peak_time_indices = np.array(peak_time_indices)
off_peak_time_indices = np.setdiff1d(all_time, peak_time_indices)

# 提取并展平三种场景的 OD 数据
od_all_flat = od.reshape(-1, od.shape[1]*od.shape[2]).flatten()
od_peak_flat = od[peak_time_indices].reshape(-1, od.shape[1]*od.shape[2]).flatten()
od_off_peak_flat = od[off_peak_time_indices].reshape(-1, od.shape[1]*od.shape[2]).flatten()


# 统计区间与标签
bins = [0,1,2,3,4,5,6,7,8,9]
labels = ['0','1','2','3','4','5','6','7','8','>=9']

# 计算各区间占比
def calc_percentages(data):
    counts = [np.sum(data==i) for i in range(9)] + [np.sum(data>=9)]
    return [c/len(data)*100 for c in counts]

all_pct = calc_percentages(od_all_flat)
peak_pct = calc_percentages(od_peak_flat)
off_peak_pct = calc_percentages(od_off_peak_flat)


# 绘图配置
plt.figure(figsize=(9,5))  # 保持原图比例
x = np.arange(len(labels))
bar_width = 0.25
colors = ['#FF6347','#4682B4','#90EE90']  # 贴近原图配色
legends = ['All Time Steps','Peak Hours','Off-Peak Hours']

# 绘制柱状图：关键！通过 align='edge' + 偏移实现刻度贴柱子右侧
# 全部时间步柱子：左边缘对齐 x - bar_width，实现视觉上右移
plt.bar(x - bar_width, all_pct, width=bar_width, color=colors[0], label=legends[0], align='edge')
# 高峰期柱子：左边缘对齐 x，自然右移
plt.bar(x, peak_pct, width=bar_width, color=colors[1], label=legends[1], align='edge')
# 平峰期柱子：左边缘对齐 x + bar_width，自然右移
plt.bar(x + bar_width, off_peak_pct, width=bar_width, color=colors[2], label=legends[2], align='edge')


# 坐标轴强化：刻度线向内 + 加粗
ax = plt.gca()
# 刻度线长度 10、宽度 1.5、方向向内
ax.tick_params(axis='both', length=10, width=2, direction='in')
# 坐标轴边框加粗（2 磅）
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 设置刻度位置和标签
ax.set_xticks(x + 2*bar_width)
ax.set_xticklabels(labels, fontsize=12)  # 设置x轴刻度文字大小

# 关键：强制设置y轴为百分比格式（两种方式确保生效）
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))  # 方式1：百分比格式化
plt.ylabel('Percentage', fontsize=18)  # 方式2：标签明确标注%




# 标签与图例（英文）
plt.xlabel('O-D flows (# of trips/time interval)', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig("./可视化/结果图/OD值稀疏分布图.pdf", format='pdf', dpi=300, bbox_inches='tight')#