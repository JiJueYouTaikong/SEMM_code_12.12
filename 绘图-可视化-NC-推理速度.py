import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

# 1. 数据准备
models = [
    'SUE-GB', 'SSM', 'Deep Gravity',
    'SAE-FC', 'GPT2', 'DeepSeek-MOE', 'Ours'
]
infer_time = [3305.414, 9358.6992, 0.0045, 0.0028, 0.0481, 0.0544, 0.0017]

# 2. 7种高区分度科研配色
colors = [
    '#a80326',  # 深红
    '#f67948',  # 橙黄
    '#fdb96b',  # 浅黄
    '#fee99d',  # 浅蓝
    '#92c5de',  # 深蓝
    '#5c90c2',  # 天青蓝
    '#3951a2'   # 藏青
]
# 2. 7种高区分度科研配色
# colors = [
#     '#a80326',  # 深红
#     '#f67948',  # 橙黄
#     '#fdb96b',  # 浅黄
#     '#f2b6c0',  # 浅蓝
#     '#cbc7d8',  # 深蓝
#     '#8db7d2',  # 天青蓝
#     '#5e62a9'   # 藏青
# ]

# 3. 绘图设置（加宽画布适配7列图例）
plt.figure(figsize=(10, 6), dpi=300)
bar_width = 0.6
x = np.arange(len(models))

# 4. 绘制柱状图（逐个绘制，绑定图例手柄）
handles = []  # 存储每个柱子的图例手柄
for i in range(len(models)):
    bar = plt.bar(
        x[i], infer_time[i],
        width=bar_width,
        color=colors[i],
        edgecolor='black',
        linewidth=1.2,
        label=models[i]  # 每个柱子单独绑定标签
        ,zorder = 3  # 关键：设置柱状图图层顺序高于网格线（网格线zorder=2）
    )
    handles.append(bar[0])  # 收集每个柱子的手柄

# 5. 核心风格调整
plt.yscale('log')
# 去除y轴小刻度
plt.minorticks_on()
plt.tick_params(axis='y', which='minor', left=False)

# 轴标签（英文+加粗，Arial字体）
plt.ylabel('Inference Time (s)', fontsize=12, fontweight='bold', fontfamily='Arial')

# 去除x轴默认数字序号
plt.xticks(ticks=x, labels=['']*len(models))

# 取消网格线 + 完整外边框
ax = plt.gca()
ax.grid(True, zorder=1,alpha=0.5)
# 统一边框样式
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.4)
    spine.set_color('black')

# 调整后的图例设置
legend = plt.legend(
    handles=handles,
    labels=models,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.07), # 尝试将图例稍微下移
    ncol=7,
    fontsize=11,
    title_fontsize=10,
    frameon=True,
    columnspacing=1.2,
    handletextpad=0.5
)

# ========== 关键修改：去除图例中每个手柄的黑边框 ==========
for handle in legend.legendHandles:
    handle.set_edgecolor('none')  # 图例手柄无边框
    handle.set_linewidth(0)       # 确保边框线宽为0（双重保险）

# 调整后的布局设置
plt.tight_layout(rect=[0, 0, 1, 1])  # 减少顶部预留的空间

# 保存为PDF格式（矢量图，支持无损缩放，适合学术论文）
# bbox_inches='tight' 确保图例不被裁剪，pad_inches=0.1 保留少量边距
plt.savefig(
    f'./可视化/结果图/NC-1-infer-time-{formatted_time}.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    facecolor='white',  # 背景色为白色（避免透明背景在论文中显示异常）
    edgecolor='none'
)

plt.close()  # 关闭画布，释放内存