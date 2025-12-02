import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import TileLayer
from shapely.geometry import Polygon
from shapely import wkt
from shapely.ops import unary_union


def grid_style(feature):
    return {
        'color': 'black',
        'weight': 1.5,
        'fillOpacity': 0
    }


# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
boundary_csv = "ad_county.csv"
time_step = 0

# 三组 OD 数据：真实值和两个预测值
od_sources = {
    "真实值": "真实值.npy",
    "Pred_ours": "../测试集TNN/Pred_RED-5.7127.npy"
}

# ---------- 加载数据 ----------
df = pd.read_csv(grid_csv)
v1 = np.load(v1_file).astype(int)
boundary_df = pd.read_csv(boundary_csv)
boundaries = [wkt.loads(w) for w in boundary_df['wkt']]
full_boundary = unary_union(boundaries)


# ---------- 颜色映射 ----------

# 分段的Reds色系映射
def get_color_map(values, cmap_name='Reds'):
    # 定义分段区间和对应的颜色
    color_bins = [
        (0, 10, '#fdcebb'),
        (10, 50, '#fcb89e'),
        (50, 100, '#fc9373'),
        (100, 500, '#db2824'),
        (500, float('inf'), '#67000d')
    ]

    colors = []
    for val in values:
        if val <= 0:
            colors.append("None")
            continue

        # 根据值查找对应的颜色区间
        for lower, upper, color in color_bins:
            if lower <= val < upper:
                colors.append(color)
                break
        else:
            # 如果值大于最大区间的上限
            colors.append('#67000d')

    return colors


# ---------- 可视化函数 ----------
def draw_map(colors, title, output_file):
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13,zoom_control = False,
scrollWheelZoom = False,
control_scale = False,
attribution_control = False)

    # TileLayer('cartodb positron').add_to(m)

    TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='CartoDB Positron No Labels',
        control=False
    ).add_to(m)

    for idx, row in df.iterrows():
        # 当前网格Polygon
        grid_poly = Polygon([
            (row['Min Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Max Latitude']),
            (row['Min Longitude'], row['Max Latitude'])
        ])

        # 与边界进行空间裁剪
        clipped_poly = full_boundary.intersection(grid_poly)

        if not clipped_poly.is_empty and clipped_poly.is_valid:
            def polygon_to_coords(polygon):
                if polygon.geom_type == 'Polygon':
                    return [[lat, lon] for lon, lat in polygon.exterior.coords]
                elif polygon.geom_type == 'MultiPolygon':
                    coords = []
                    for part in polygon.geoms:
                        coords.append([[lat, lon] for lon, lat in part.exterior.coords])
                    return coords
                else:
                    return []

            coords = polygon_to_coords(clipped_poly)
            if isinstance(coords[0][0], list):  # MultiPolygon
                for part in coords:
                    folium.Polygon(
                        locations=part,
                        color='black',
                        weight=1.0,
                        fill=True,
                        fill_color=colors[idx],
                        fill_opacity=1
                    ).add_to(m)
            else:  # Polygon
                folium.Polygon(
                    locations=coords,
                    color='black',
                    weight=1.0,
                    fill=True,
                    fill_color=colors[idx],
                    fill_opacity=1
                ).add_to(m)

    m.save(output_file)
    print(f"✅ {title} 地图已保存为 {output_file}")


# ---------- 主循环处理所有 OD 源 ----------
for label, file_path in od_sources.items():
    od_matrix = np.load(file_path)[time_step]  # [110, 110]

    # 统计出发量和到达量
    total_departure = np.sum(od_matrix, axis=1)
    total_arrival = np.sum(od_matrix, axis=0)

    # 映射到 1.1km 网格
    mapped_departure = np.zeros(len(df))
    mapped_arrival = np.zeros(len(df))
    for i, idx in enumerate(v1):
        mapped_departure[idx] += total_departure[i]
        mapped_arrival[idx] += total_arrival[i]

    # 映射颜色
    departure_colors = get_color_map(mapped_departure)
    arrival_colors = get_color_map(mapped_arrival)

    # 输出地图
    draw_map(departure_colors, f"{label}-总出发量", f"../案例分析/{label}_总出发量.html")
    draw_map(arrival_colors, f"{label}-总到达量", f"../案例分析/{label}-总到达量.html")