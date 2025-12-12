import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import TileLayer

# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
od_file = "真实值.npy"
time_step = 0

# ---------- 加载数据 ----------
df = pd.read_csv(grid_csv, sep=',')
v1 = np.load(v1_file).astype(int)  # 110个映射索引
od_matrix = np.load(od_file)[time_step]  # [110, 110]

# ---------- 统计总出发/到达量 ----------
total_departure = np.sum(od_matrix, axis=1)
total_arrival = np.sum(od_matrix, axis=0)

# ---------- 映射至网格 ----------
mapped_departure = np.zeros(len(df))
mapped_arrival = np.zeros(len(df))
for i, idx in enumerate(v1):
    mapped_departure[idx] += total_departure[i]
    mapped_arrival[idx] += total_arrival[i]

# ---------- 统计110个区域的边界范围 ----------
used_df = df.iloc[v1]
lat_min = used_df[['Min Latitude', 'Max Latitude']].min().min()
lat_max = used_df[['Min Latitude', 'Max Latitude']].max().max()
lon_min = used_df[['Min Longitude', 'Max Longitude']].min().min()
lon_max = used_df[['Min Longitude', 'Max Longitude']].max().max()

# ---------- 颜色映射函数 ----------
def get_color_map(values, cmap_name='YlOrRd'):
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(norm(val))) if val > 0 else "#ffffff" for val in values]

departure_colors = get_color_map(mapped_departure)
arrival_colors = get_color_map(mapped_arrival)

# ---------- 地图绘制函数（限定矩形区域） ----------
def draw_map(colors, title, output_file):
    m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=13)
    TileLayer('cartodb positron').add_to(m)

    for idx, row in df.iterrows():
        # 跳过不在目标矩形内的网格
        if (row['Min Latitude'] > lat_max or row['Max Latitude'] < lat_min or
            row['Min Longitude'] > lon_max or row['Max Longitude'] < lon_min):
            continue

        corners = [
            [row['Min Latitude'], row['Min Longitude']],
            [row['Min Latitude'], row['Max Longitude']],
            [row['Max Latitude'], row['Max Longitude']],
            [row['Max Latitude'], row['Min Longitude']]
        ]
        folium.Polygon(
            locations=corners,
            color='black',
            weight=1,
            fill=True,
            fill_color=colors[idx],
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_file)
    print(f"✅ {title} 地图已保存为 {output_file}")

# ---------- 绘制两个地图 ----------
draw_map(departure_colors, "总出发量", "departure_map_clipped.html")
draw_map(arrival_colors, "总到达量", "arrival_map_clipped.html")
