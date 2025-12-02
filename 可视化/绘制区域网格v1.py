import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
# 读取 CSV 文件（自动判断分隔符）
with open('1.1km网格.csv', encoding='utf-8') as f:
    first_line = f.readline()
sep = '\t' if '\t' in first_line else ','

df = pd.read_csv('1.1km网格.csv', sep=sep)

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 8))

# 遍历绘制每个区域为矩形框（无填充，黑色边框）
for idx, row in df.iterrows():
    width = row['Max Longitude'] - row['Min Longitude']
    height = row['Max Latitude'] - row['Min Latitude']

    rect = patches.Rectangle(
        (row['Min Longitude'], row['Min Latitude']),
        width, height,
        linewidth=0.8,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(rect)

# 可选：加上中心点文字
# for idx, row in df.iterrows():
#     ax.text(row['Center Longitude'], row['Center Latitude'],
#             str(int(row['Region ID'])),
#             fontsize=6, ha='center', va='center')

# 设置坐标轴标签和标题
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('City Grid Map (1.1km x 1.1km)', fontsize=13)

# 设置坐标轴范围
padding = 0.001
ax.set_xlim(df['Min Longitude'].min() - padding, df['Max Longitude'].max() + padding)
ax.set_ylim(df['Min Latitude'].min() - padding, df['Max Latitude'].max() + padding)

# 去除背景格线和边框
ax.set_aspect('equal', adjustable='box')
ax.grid(False)
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)  # 仅可选去除上右边框

# 去除 seaborn 风格背景
plt.style.use('default')  # 或 plt.style.use('classic')

plt.tight_layout()
plt.show()
