import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, Point
from shapely import wkt
from shapely.ops import unary_union
import math
import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# ---------- 参数配置 ----------
grid_csv = "../地图/1.1km网格.csv"
v1_file = "../地图/网格映射v1.npy"
boundary_csv = "../地图/ad_county.csv"
time_step = 9  # 第10个时间步(0-based)

# 三组 OD 数据：真实值和两个预测值
od_sources = {
    "真实值": "../测试集TNN/真实值.npy",
    "Pred_ours": "../测试集TNN/Pred_RED-5.7127.npy"
}

# 地图中心坐标
center_lat, center_lon = 30.4341, 114.5113
# 图片尺寸和分辨率
fig_size = (15,8)
dpi = 300

# ---------- 加载数据 ----------
# 1. 网格数据
df = pd.read_csv(grid_csv)
# 构建网格的几何对象
df['geometry'] = df.apply(lambda row: Polygon([
    (row['Min Longitude'], row['Min Latitude']),
    (row['Max Longitude'], row['Min Latitude']),
    (row['Max Longitude'], row['Max Latitude']),
    (row['Min Longitude'], row['Max Latitude'])
]), axis=1)
gdf_grid = gpd.GeoDataFrame(df, geometry='geometry')
gdf_grid.crs = "EPSG:4326"  # WGS84坐标系

# 2. 网格映射
v1 = np.load(v1_file).astype(int)
v1_set = set(v1)

# 3. 行政边界
boundary_df = pd.read_csv(boundary_csv)
boundaries = [wkt.loads(w) for w in boundary_df['wkt']]
full_boundary = unary_union(boundaries)

# ---------- 辅助函数 ----------
# 颜色映射函数
def get_color_map(values, cmap_name='Accent'):
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(cmap_name)
    return [cmap(norm(val)) if val > 0 else (1, 1, 1, 0) for val in values]

# 生成贝塞尔弧线坐标（用于Matplotlib绘制）
def generate_arc_coordinates(start, end, resolution=10):
    lat1, lon1 = start
    lat2, lon2 = end

    # 中点
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    # 距离和弧线高度
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    distance = math.sqrt(dlat ** 2 + dlon ** 2)
    height = distance * 0.2

    # 控制点
    perp_lat = -(lon2 - lon1) * 0.1
    perp_lon = (lat2 - lat1) * 0.1
    control_lat = mid_lat + perp_lat
    control_lon = mid_lon + perp_lon

    # 贝塞尔曲线点
    points = []
    for i in range(resolution + 1):
        t = i / resolution
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t ** 2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * control_lon + t ** 2 * lon2
        points.append((lon, lat))  # Matplotlib是(lon, lat)顺序
    return points

# 绘制单个网格的几何形状（处理裁剪后的多边形）
def plot_clipped_polygon(ax, poly, fill_color, edge_color='black', edge_width=2):
    if poly.geom_type == 'Polygon':
        # 提取外部坐标
        coords = list(poly.exterior.coords)
        mpl_poly = MplPolygon(coords, closed=True, facecolor=fill_color,
                              edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(mpl_poly)
    elif poly.geom_type == 'MultiPolygon':
        for part in poly.geoms:
            coords = list(part.exterior.coords)
            mpl_poly = MplPolygon(coords, closed=True, facecolor=fill_color,
                                  edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(mpl_poly)

# ---------- 主可视化函数 ----------
def plot_od_map(od_matrix, label, output_format='png'):
    # 1. 初始化画布
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xlim(gdf_grid['Min Longitude'].min(), gdf_grid['Max Longitude'].max())
    ax.set_ylim(gdf_grid['Min Latitude'].min(), gdf_grid['Max Latitude'].max())
    ax.axis('off')  # 隐藏坐标轴

    # 2. 计算网格中心点并绘制网格
    grid_centers = {}
    for idx, row in gdf_grid.iterrows():
        # 空间裁剪
        clipped_poly = full_boundary.intersection(row['geometry'])
        if clipped_poly.is_empty or not clipped_poly.is_valid:
            continue

        # 计算中心点
        center_lat = (row['Min Latitude'] + row['Max Latitude']) / 2
        center_lon = (row['Min Longitude'] + row['Max Longitude']) / 2
        grid_centers[idx] = (center_lat, center_lon)

        # 填充颜色（仅v1区域）
        fill_color = (1, 1, 1, 0) if idx not in v1_set else (0.8, 0.8, 0.8, 0.7)  # 灰色填充示例
        # 绘制裁剪后的网格
        plot_clipped_polygon(ax, clipped_poly, fill_color=fill_color, edge_width=1)

    # 3. 绘制OD弧线
    for origin_idx in grid_centers:
        if origin_idx not in v1_set:
            continue
        origin_center = grid_centers[origin_idx]
        for dest_idx in grid_centers:
            if dest_idx == origin_idx or dest_idx not in v1_set:
                continue

            # 获取OD值
            orig_od_idx = np.where(v1 == origin_idx)[0][0]
            dest_od_idx = np.where(v1 == dest_idx)[0][0]
            od_value = od_matrix[orig_od_idx, dest_od_idx]

            if od_value >= 1:
                dest_center = grid_centers[dest_idx]
                # 生成弧线坐标
                arc_coords = generate_arc_coordinates(origin_center, dest_center)
                # 根据OD值设置样式
                if od_value <= 5:
                    color = '#5e62a9'
                    linewidth = 1.3
                    alpha = 0.4
                elif od_value <= 10:
                    color = '#fdffb6'
                    linewidth = 1.4
                    alpha = 0.75
                else:
                    color = '#93002e'
                    linewidth = 1.5
                    alpha = 1.0
                # 绘制弧线
                ax.plot([p[0] for p in arc_coords], [p[1] for p in arc_coords],
                        color=color, linewidth=linewidth, alpha=alpha, zorder=10)

    # 4. 保存图片
    plt.tight_layout()
    output_file = f"{label}_od_all_origins.{output_format}"
    plt.savefig(output_file, format=output_format, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ {label} 地图已保存为 {output_file}")

# ---------- 遍历OD数据绘制 ----------
for label, file_path in od_sources.items():
    od_matrix = np.load(file_path)[time_step]  # [110, 110]
    # 绘制并保存为PNG（可改为pdf）
    plot_od_map(od_matrix, label, output_format='png')
    # 若需PDF，取消下面注释
    # plot_od_map(od_matrix, label, output_format='pdf')