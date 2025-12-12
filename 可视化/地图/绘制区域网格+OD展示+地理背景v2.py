import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import TileLayer

# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
od_file = "Pred_ANN.npy"
time_step = 0

# ---------- 加载数据 ----------
df = pd.read_csv(grid_csv, sep=',')
v1 = np.load(v1_file)  # 长度 110
od_matrix = np.load(od_file)[time_step]  # [110, 110]

# ---------- 统计出发量和到达量 ----------
total_departure = np.sum(od_matrix, axis=1)  # shape [110]
total_arrival = np.sum(od_matrix, axis=0)  # shape [110]

# ---------- 构建映射关系 ----------
v1 = v1.astype(int)
mapped_departure = np.zeros(len(df))
mapped_arrival = np.zeros(len(df))
for i, idx in enumerate(v1):
    mapped_departure[idx] += total_departure[i]
    mapped_arrival[idx] += total_arrival[i]


# ---------- 颜色映射函数 ----------
def get_color_map(values, cmap_name='YlOrRd'):
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(cmap_name)
    colors = [mcolors.to_hex(cmap(norm(val))) if val > 0 else "#ffffff" for val in values]
    return colors


departure_colors = get_color_map(mapped_departure)
arrival_colors = get_color_map(mapped_arrival)


# ---------- 可视化函数 ----------
def draw_map(colors, title, output_file):
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13)
    TileLayer('cartodb positron').add_to(m)

    for idx, row in df.iterrows():
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


# ---------- 生成两张地图 ----------
draw_map(departure_colors, "总出发量", "departure_map_pred_ANN.html")
draw_map(arrival_colors, "总到达量", "arrival_map_pred_ANN.html")
