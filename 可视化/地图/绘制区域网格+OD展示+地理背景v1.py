import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import GeoJson, TileLayer

# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
od_file = "真实值.npy"
time_step = 0  # 要可视化的时间步
max_lines = 500  # 控制最多画的OD线数量（避免地图太密）

# ---------- 加载数据 ----------
df = pd.read_csv(grid_csv, sep=',')  # 若分隔符不是tab，根据文件调整
v1 = np.load(v1_file)  # 形状 [110]
od_matrix = np.load(od_file)[time_step]  # [110, 110]

# ---------- 初始化地图 ----------
center_lat = 30.434140477236866
center_lon = 114.51135311054612
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
TileLayer('cartodb positron').add_to(m)  # 或者使用任意你想要的底图
# ---------- 添加网格框 ----------
for idx, row in df.iterrows():
    corners = [
        [row['Min Latitude'], row['Min Longitude']],
        [row['Min Latitude'], row['Max Longitude']],
        [row['Max Latitude'], row['Max Longitude']],
        [row['Max Latitude'], row['Min Longitude']],
        [row['Min Latitude'], row['Min Longitude']]  # 闭合
    ]
    folium.PolyLine(locations=corners, color="black", weight=2, opacity=0.6).add_to(m)

# ---------- 构建流量颜色映射 ----------
norm = mcolors.Normalize(vmin=od_matrix.min(), vmax=od_matrix.max())
cmap = cm.get_cmap('autumn')  # 黄色→红色映射

# ---------- 画 OD 流线 ----------
lines = []
for i in range(110):
    for j in range(110):
        val = od_matrix[i, j]
        if val <= 0: continue
        src_idx = int(v1[i])
        dst_idx = int(v1[j])
        # print(type(src_idx), type(dst_idx))
        src_lat, src_lon = df.iloc[src_idx][['Center Latitude', 'Center Longitude']]
        dst_lat, dst_lon = df.iloc[dst_idx][['Center Latitude', 'Center Longitude']]
        color = mcolors.to_hex(cmap(norm(val)))
        lines.append((val, [(src_lat, src_lon), (dst_lat, dst_lon)], color))

# 只画 top-k 最强流量
lines.sort(reverse=True, key=lambda x: x[0])
for _, points, color in lines[:max_lines]:
    folium.PolyLine(locations=points, color=color, weight=2, opacity=0.7).add_to(m)

# ---------- 保存并展示 ----------
m.save("od_map_folium.html")
print("✅ 地图已保存为 od_map_folium.html，请用浏览器打开查看。")
