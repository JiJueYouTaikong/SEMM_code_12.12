import folium
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import TileLayer, PolyLine
from shapely.geometry import Polygon
from shapely import wkt
from shapely.ops import unary_union
import math

def grid_style(feature):
    return {'color': 'black', 'weight': 3, 'fillOpacity': 0}

# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
boundary_csv = "ad_county.csv"
time_step = 9

od_sources = {
    "真实值": "../测试集TNN/真实值.npy",
    "Pred_ours": "../测试集TNN/Pred_RED-5.7127.npy"
}

# ---------- 数据加载 ----------
df = pd.read_csv(grid_csv)
v1 = np.load(v1_file).astype(int)
boundary_df = pd.read_csv(boundary_csv)
boundaries = [wkt.loads(w) for w in boundary_df['wkt']]
full_boundary = unary_union(boundaries)

# ---------- 颜色映射 ----------
def map_color(val):
    if val <= 5:
        return '#5e62a9'
    elif val <= 10:
        return '#fdffb6'
    else:
        return '#c45161'

# 生成弧线

def generate_arc_coordinates(start, end, resolution=10):
    lat1, lon1 = start
    lat2, lon2 = end
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance = math.sqrt(dlat ** 2 + dlon ** 2)
    height = distance * 0.2
    perp_lat = -(lon2 - lon1) * 0.1
    perp_lon = (lat2 - lat1) * 0.1
    control_lat = mid_lat + perp_lat
    control_lon = mid_lon + perp_lon
    points = []
    for i in range(resolution + 1):
        t = i / resolution
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        points.append([lat, lon])
    return points

# ---------- 可视化 ----------
def draw_map(title, output_file, od_matrix=None, origin_idx=None, v1=None, df=None):
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13)
    TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='CartoDB', name='CartoDB', control=False).add_to(m)

    grid_centers = {}
    fill_colors = ["#ffffff"] * len(df)

    # 第一步：得到网格中心点
    for idx, row in df.iterrows():
        grid_poly = Polygon([
            (row['Min Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Max Latitude']),
            (row['Min Longitude'], row['Max Latitude'])])
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
            center_lat = (row['Min Latitude'] + row['Max Latitude']) / 2
            center_lon = (row['Min Longitude'] + row['Max Longitude']) / 2
            grid_centers[idx] = (center_lat, center_lon)

    # 第二步：绘制OD弧线并填充结束网格
    if od_matrix is not None and origin_idx is not None and origin_idx in grid_centers:
        origin_center = grid_centers[origin_idx]
        for dest_idx in grid_centers:
            if dest_idx != origin_idx:
                if origin_idx in v1 and dest_idx in v1:
                    orig_idx_od = np.where(v1 == origin_idx)[0][0]
                    dest_idx_od = np.where(v1 == dest_idx)[0][0]
                    od_value = od_matrix[orig_idx_od, dest_idx_od]
                else:
                    od_value = 0

                if od_value > 0:
                    dest_center = grid_centers[dest_idx]
                    arc_coords = generate_arc_coordinates(origin_center, dest_center)
                    line_color = map_color(od_value)

                    folium.PolyLine(
                        locations=arc_coords,
                        color=line_color,
                        weight=7,
                        opacity=1,
                        # tooltip=f"OD值: {od_value:.0f}"
                    ).add_to(m)

                    # 设置填色
                    fill_colors[dest_idx] = line_color

        # # 标记出发区域
        # folium.Marker(
        #     location=origin_center,
        #     icon=folium.Icon(color='red', icon='circle', prefix='fa'),
        #     tooltip=f"出发量最大区域: ID{origin_idx}"
        # ).add_to(m)

    # 最后填色网格
    for idx, row in df.iterrows():
        grid_poly = Polygon([
            (row['Min Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Max Latitude']),
            (row['Min Longitude'], row['Max Latitude'])])
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
            if isinstance(coords[0][0], list):
                for part in coords:
                    folium.Polygon(
                        locations=part,
                        color='black',
                        weight=5,
                        fill=True,
                        fill_color=fill_colors[idx],
                        fill_opacity=0.7 if idx in v1 else 0
                    ).add_to(m)
            else:
                folium.Polygon(
                    locations=coords,
                    color='black',
                    weight=5,
                    fill=True,
                    fill_color=fill_colors[idx],
                    fill_opacity=0.7 if idx in v1 else 0
                ).add_to(m)

    m.save(output_file)
    print(f"\u2705 {title} 地图已保存为 {output_file}")

# ---------- 主调用 ----------
true_od = np.load(od_sources["\u771f实值"])[time_step]
total_departure_true = np.sum(true_od, axis=1)
mapped_departure_true = np.zeros(len(df))
for i, idx in enumerate(v1):
    mapped_departure_true[idx] += total_departure_true[i]
max_departure_idx = np.argmax(mapped_departure_true)

for label, file_path in od_sources.items():
    od_matrix = np.load(file_path)[time_step]
    draw_map(
        title=f"{label} - OD流动(区域ID{max_departure_idx}出发)",
        output_file=f"{label}_od_flow_map.html",
        od_matrix=od_matrix,
        origin_idx=max_departure_idx,
        v1=v1,
        df=df
    )