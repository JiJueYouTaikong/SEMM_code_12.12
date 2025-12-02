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
    return {
        'color': 'black',
        'weight': 5,  # 加粗边界
        'fillOpacity': 0
    }


# ---------- 参数 ----------
grid_csv = "1.1km网格.csv"
v1_file = "网格映射v1.npy"
boundary_csv = "ad_county.csv"
time_step = 9  # 第10个时间步(0-based)

# 三组 OD 数据：真实值和两个预测值
od_sources = {
    "真实值": "../测试集TNN/真实值.npy",
    "Pred_ours": "../测试集TNN/Pred_RED-5.7127.npy"
}

# ---------- 加载数据 ----------
df = pd.read_csv(grid_csv)
v1 = np.load(v1_file).astype(int)
boundary_df = pd.read_csv(boundary_csv)
boundaries = [wkt.loads(w) for w in boundary_df['wkt']]
full_boundary = unary_union(boundaries)


# ---------- 颜色映射 ----------
def get_color_map(values, cmap_name='Accent'):
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(norm(val))) if val > 0 else "#ffffff" for val in values]


# 生成弧线坐标
def generate_arc_coordinates(start, end, resolution=10):
    """生成两点之间的弧线坐标"""
    lat1, lon1 = start
    lat2, lon2 = end

    # 计算中点
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    # 计算距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance = math.sqrt(dlat ** 2 + dlon ** 2)

    # 弧线高度(与距离相关)
    height = distance * 0.2

    # 垂直于线段的方向
    perp_lat = -(lon2 - lon1) * 0.1
    perp_lon = (lat2 - lat1) * 0.1

    # 控制点位置
    control_lat = mid_lat + perp_lat
    control_lon = mid_lon + perp_lon

    # 生成贝塞尔曲线点
    points = []
    for i in range(resolution + 1):
        t = i / resolution
        # 贝塞尔曲线公式
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        points.append([lat, lon])

    return points


# ---------- 可视化函数 ----------
def draw_map(colors, title, output_file, od_matrix=None, origin_idx=None, v1=None, df=None):
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13,zoom_control = False,
scrollWheelZoom = False,
control_scale = False,
attribution_control = False)

    TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='CartoDB Positron No Labels',
        control=False
    ).add_to(m)

    # 存储网格中心点
    grid_centers = {}

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

            # 计算网格中心点
            center_lat = (row['Min Latitude'] + row['Max Latitude']) / 2
            center_lon = (row['Min Longitude'] + row['Max Longitude']) / 2
            grid_centers[idx] = (center_lat, center_lon)

            # 检查是否为110个区域
            fill_color = colors[idx] if idx in v1 else "#ffffff"

            if isinstance(coords[0][0], list):  # MultiPolygon
                for part in coords:
                    folium.Polygon(
                        locations=part,
                        color='black',
                        weight=5,  # 加粗边界
                        fill=True,
                        fill_color=fill_color,
                        fill_opacity=0.7 if idx in v1 else 0  # 非110个区域不填充
                    ).add_to(m)
            else:  # Polygon
                folium.Polygon(
                    locations=coords,
                    color='black',
                    weight=5,  # 加粗边界
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.7 if idx in v1 else 0  # 非110个区域不填充
                ).add_to(m)

    # 绘制OD弧线
    if od_matrix is not None and origin_idx is not None and origin_idx in grid_centers:
        origin_center = grid_centers[origin_idx]

        # 绘制从出发区域到其他所有区域的OD弧线
        for dest_idx in grid_centers:
            if dest_idx != origin_idx:
                if origin_idx in v1 and dest_idx in v1:
                    orig_idx_od = np.where(v1 == origin_idx)[0][0]
                    dest_idx_od = np.where(v1 == dest_idx)[0][0]
                    od_value = od_matrix[orig_idx_od, dest_idx_od]
                else:
                    od_value = 0

                if od_value >= 1:
                    dest_center = grid_centers[dest_idx]
                    arc_coords = generate_arc_coordinates(origin_center, dest_center)

                    # 按区间设置颜色
                    if od_value <= 5:
                        line_color = '#5e62a9'
                    elif od_value <= 10:
                        line_color = '#fdffb6'
                    else:
                        line_color = '#93002e'

                    # 固定线宽为10
                    folium.PolyLine(
                        locations=arc_coords,
                        color=line_color,
                        weight=8,
                        opacity=1,
                        tooltip=f"OD值: {od_value:.2f}"
                    ).add_to(m)

        # # 标记出发区域
        # folium.Marker(
        #     location=origin_center,
        #     icon=folium.Icon(color='red', icon='circle', prefix='fa'),
        #     tooltip=f"出发量最大区域: ID{origin_idx} (出发量:{mapped_departure[origin_idx]:.0f})"
        # ).add_to(m)

    m.save(output_file)
    print(f"✅ {title} 地图已保存为 {output_file}")


# ---------- 主循环：对每组 OD 数据绘图（每个区域都作为出发点） ----------
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

    # 初始化地图对象
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13,
                   zoom_control=False, scrollWheelZoom=False,
                   control_scale=False, attribution_control=False)

    TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='CartoDB', name='CartoDB Positron No Labels', control=False
    ).add_to(m)

    # 预绘网格区域（只做一次）
    grid_centers = {}
    for idx, row in df.iterrows():
        grid_poly = Polygon([
            (row['Min Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Min Latitude']),
            (row['Max Longitude'], row['Max Latitude']),
            (row['Min Longitude'], row['Max Latitude'])
        ])
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

            fill_color = "#ffffff"
            if isinstance(coords[0][0], list):  # MultiPolygon
                for part in coords:
                    folium.Polygon(locations=part, color='black', weight=2,
                                   fill=True, fill_color=fill_color, fill_opacity=0).add_to(m)
            else:
                folium.Polygon(locations=coords, color='black', weight=2,
                               fill=True, fill_color=fill_color, fill_opacity=0).add_to(m)

    # 绘制所有出发区域的弧线
    for origin_idx in grid_centers:
        if origin_idx in v1:
            for dest_idx in grid_centers:
                if dest_idx == origin_idx:
                    continue
                if dest_idx in v1:
                    orig_od_idx = np.where(v1 == origin_idx)[0][0]
                    dest_od_idx = np.where(v1 == dest_idx)[0][0]
                    od_value = od_matrix[orig_od_idx, dest_od_idx]
                else:
                    od_value = 0

                if od_value >= 1:
                    origin_center = grid_centers[origin_idx]
                    dest_center = grid_centers[dest_idx]
                    arc_coords = generate_arc_coordinates(origin_center, dest_center)

                    # 颜色选择
                    if od_value <= 5:
                        line_color = '#402d94'
                        line_weight = 4
                        line_opacity = 0.4
                    elif od_value <= 10:
                        line_color = '#fdffb6'
                        line_weight = 5
                        line_opacity = 0.75
                    else:
                        line_color = '#93002e'
                        line_weight = 6
                        line_opacity = 1.0

                    folium.PolyLine(
                        locations=arc_coords,
                        color=line_color,
                        weight=line_weight,  # 线条变细
                        opacity=line_opacity,  # 半透明，便于多个线重叠观察
                        tooltip=f"{origin_idx} → {dest_idx}: {od_value:.1f}"
                    ).add_to(m)

    # 保存地图
    output_file = f"{label}_od_all_origins.html"
    m.save(output_file)
    print(f"✅ {label} 地图已保存为 {output_file}")