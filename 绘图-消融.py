import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置英文字体为 Arial，中文为 SimHei 或其他
rcParams['font.family'] = 'Arial'

# 模型名称与颜色
models = ["w/o Temp", "w/o Freq", "w/o Random", "RE"]
colors = ["#f2b6c0", "#cbc7d8", "#8db7d2", "#5e62a9"]

# 指标数据
metrics = {
    "RMSE": [6.1294, 6.0857, 5.9714, 5.7127],
    "MAE":  [0.8798, 0.8961, 0.8455, 0.8276],
    "CPC":  [0.6646, 0.6589, 0.6727, 0.6807],
    "JSD":  [0.1093, 0.1111, 0.1066, 0.1049]
}
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
    ax.grid(False)

    # 设置Y轴范围留出空隙便于视觉对比
    val_range = max(values) - min(values)
    margin = val_range * 0.25 if val_range != 0 else 0.1
    ax.set_ylim(min(values) - margin, max(values) + margin)

# 布局优化
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
# 保存为高清图片
# plt.savefig("可视化/消融-敏感性/消融-RE柱状图.png", bbox_inches="tight", dpi=600)
plt.savefig("可视化/消融-敏感性/消融-RE柱状图.pdf", format='pdf', bbox_inches="tight", dpi=300)
plt.show()
