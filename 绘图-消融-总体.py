import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置英文字体为 Arial，中文为 SimHei 或其他
rcParams['font.family'] = 'Arial'

# 模型名称（更新为新信息）与颜色（保持4个配色适配新模型数量）
models = ["w/o MCSM", "w/o SC", "w/o RE", "SEMM"]
# colors = ["#f2b6c0", "#cbc7d8", "#8db7d2", "#5e62a9"]
colors = ["#fee99d", "#92c5de", "#5c90c2", "#3951a2"]

# 指标数据（更新为新信息）
metrics = {
    "RMSE": [5.9028, 5.8476, 6.2156, 5.7127],
    "MAE":  [0.857, 0.9856, 0.9173, 0.8276],
    "CPC":  [0.6669, 0.5903, 0.661, 0.6807],
    "JSD":  [0.1112, 0.1476, 0.1065, 0.1049]
}


plt.rcParams['axes.grid'] = True  # 全局网格开启
plt.rcParams['grid.alpha'] = 0.5  # 网格透明度
plt.rcParams['grid.linewidth'] = 0.95  # 网格线宽度
plt.rcParams['grid.color'] = '#E0E0E0'  # 网格线颜色



# 设置简洁风格
sns.set(style="white", context="paper", font_scale=1.1)

# 创建图形
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# 绘制每个指标
for i, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[i]
    bars = ax.bar(models, values, color=colors, width=0.65,edgecolor='black',
              linewidth=0.8)
    ax.set_title(metric_name, fontsize=14)
    ax.tick_params(axis='x', labelrotation=30)

    # 关闭网格线
    ax.grid(True)

    # 设置Y轴范围留出空隙便于视觉对比
    val_range = max(values) - min(values)
    margin = val_range * 0.25 if val_range != 0 else 0.1
    ax.set_ylim(min(values) - margin, max(values) + margin)

# 布局优化
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
# 保存为高清图片
# plt.savefig("可视化/消融-敏感性/消融-RE柱状图.png", bbox_inches="tight", dpi=600)
plt.savefig("./可视化/消融-敏感性/消融-总体柱状图.pdf", format='pdf', bbox_inches="tight", dpi=300)
# plt.show()
