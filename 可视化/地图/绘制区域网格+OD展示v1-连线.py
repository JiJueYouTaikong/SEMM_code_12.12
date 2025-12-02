import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm

# --- 文件路径 ---
grid_csv_path = '1.1km网格.csv'
od_path = '真实值.npy'
v1_path = '网格映射v1.npy'

# --- 读取数据 ---
with open(grid_csv_path, encoding='utf-8') as f:
    sep = '\t' if '\t' in f.readline() else ','
grid_df = pd.read_csv(grid_csv_path, sep=sep)
od_data = np.load(od_path)       # [168, 110, 110]
v1 = np.load(v1_path)            # [110] 网格ID

# --- 网格 ID → 中心经纬度映射 ---
id_to_coord = {}
for _, row in grid_df.iterrows():
    region_id = int(row['Region ID'])
    id_to_coord[region_id] = (row['Center Longitude'], row['Center Latitude'])

# --- 获取合法区域中心点坐标（顺序和v1一致） ---
region_coords = [id_to_coord[rid] for rid in v1]

# --- 选定时间步 ---
i = 0  # 可修改为 0~167
od_matrix = od_data[i]

# --- 自定义颜色映射（黄 → 橙 → 红） ---
colors = [(1, 1, 0), (1, 0.5, 0), (1, 0, 0)]  # RGB for yellow, orange, red
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# --- 准备绘图 ---
fig, ax = plt.subplots(figsize=(10, 8))

# === 1. 绘制所有网格框（不填充，黑色边框） ===
for _, row in grid_df.iterrows():
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

# === 2. OD 流动连线 ===
lines = []
weights = []

for src in range(110):
    for dst in range(110):
        if src == dst:
            continue
        value = od_matrix[src, dst]
        if value <= 0:
            continue
        start = region_coords[src]
        end = region_coords[dst]
        lines.append([start, end])
        weights.append(value)

# === 3. 绘制流线 ===
if weights:
    norm = Normalize(vmin=np.min(weights), vmax=np.max(weights))
    line_segments = LineCollection(
        lines,
        colors=[custom_cmap(norm(w)) for w in weights],
        linewidths=1.0,
        alpha=0.8
    )
    ax.add_collection(line_segments)

    # 添加颜色条
    sm = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array(weights)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('OD Flow Intensity')

# === 4. 图形美化 ===
# 设置图标题和比例
ax.set_title(f'OD Flow over City Grid (Time {i})', fontsize=14)
ax.set_aspect('equal', adjustable='box')

# 自动缩放
padding = 0.001
ax.set_xlim(grid_df['Min Longitude'].min() - padding, grid_df['Max Longitude'].max() + padding)
ax.set_ylim(grid_df['Min Latitude'].min() - padding, grid_df['Max Latitude'].max() + padding)

# 去除四条边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 去除坐标刻度
ax.set_xticks([])
ax.set_yticks([])

# 去除背景网格
ax.grid(False)
plt.tight_layout()
plt.show()
