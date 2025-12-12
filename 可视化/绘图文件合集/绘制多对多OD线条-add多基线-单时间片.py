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

# 扩展OD数据：真实值 + 5个预测方法（SUE-GB SSM GPT2 DeepGravity Ours）
od_sources = {
    "SUE-GB": "../测试集TNN/Pred-SUE-GB.npy",  # 请替换为实际文件路径
    "SSM": "../测试集TNN/Pred_SSM_SUE_MSA_6.15_7.80.npy",  # 请替换为实际文件路径
    "GPT2": "../测试集TNN/Pred_GPT2.npy",  # 请替换为实际文件路径
    "DeepGravity": "../测试集TNN/Pred_DG_false.npy",  # 请替换为实际文件路径
    "Ours": "../测试集TNN/Pred_RED-5.7127.npy",
    "Ground Truth": "../测试集TNN/真实值.npy"
}

# 子图标注（按顺序对应od_sources的键）
subplot_labels = ['(a) SUE-GB', '(b) SSM', '(c) GPT2', '(d) DeepGravity', '(e) Ours', '(f) Ground Truth']

# 地图中心坐标
center_lat, center_lon = 30.4341, 114.5113
# 图片尺寸和分辨率（调整为适应2行3列布局）
fig_size = (24, 16)  # 宽24，高16，保证子图清晰
dpi = 300

# 颜色配置（按绘制顺序：浅色、中间色、红色）
COLOR_CONFIG = {
    "light": {
        "range": (1, 5),
        "color": '#5e62a9',
        "linewidth": 1.3,
        "alpha": 1,
        "zorder": 10
    },
    "middle": {
        "range": (5, 10),
        "color": '#fdffb6',
        "linewidth": 1.4,
        "alpha": 1,
        "zorder": 11
    },
    "red": {
        "range": (10, np.inf),
        "color": '#93002e',
        "linewidth": 1.4,
        "alpha": 1,
        "zorder": 12
    }
}
# 绘制顺序：先浅色，再中间色，最后红色
DRAW_ORDER = ["light", "middle", "red"]

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
# 预建v1索引到网格idx的映射字典（提升查询效率）
v1_to_grid = {idx: grid_idx for idx, grid_idx in enumerate(v1)}
grid_to_v1 = {grid_idx: idx for idx, grid_idx in enumerate(v1)}

# 3. 行政边界
boundary_df = pd.read_csv(boundary_csv)
boundaries = [wkt.loads(w) for w in boundary_df['wkt']]
full_boundary = unary_union(boundaries)

# 预计算网格中心点和裁剪后的多边形（避免重复计算）
grid_centers = {}
clipped_polygons = {}
for idx, row in gdf_grid.iterrows():
    # 空间裁剪
    clipped_poly = full_boundary.intersection(row['geometry'])
    if clipped_poly.is_empty or not clipped_poly.is_valid:
        continue
    clipped_polygons[idx] = clipped_poly

    # 计算中心点
    center_lat = (row['Min Latitude'] + row['Max Latitude']) / 2
    center_lon = (row['Min Longitude'] + row['Max Longitude']) / 2
    grid_centers[idx] = (center_lat, center_lon)

# 预计算坐标范围
x_min, x_max = gdf_grid['Min Longitude'].min(), gdf_grid['Max Longitude'].max()
y_min, y_max = gdf_grid['Min Latitude'].min(), gdf_grid['Max Latitude'].max()


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


# ---------- 子图绘制函数 ----------
def plot_od_subplot(ax, od_matrix, label):
    # 1. 配置子图属性
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')  # 隐藏坐标轴

    # 2. 绘制网格
    for idx in clipped_polygons:
        clipped_poly = clipped_polygons[idx]
        # 填充颜色（仅v1区域）
        fill_color = (1, 1, 1, 0) if idx not in v1_set else (0.8, 0.8, 0.8, 0.7)
        # 绘制裁剪后的网格
        plot_clipped_polygon(ax, clipped_poly, fill_color=fill_color, edge_width=1)

    # 3. 分类收集OD弧线数据（按颜色类型）
    arc_data = {
        "light": [],
        "middle": [],
        "red": []
    }

    for origin_idx in grid_centers:
        if origin_idx not in v1_set:
            continue
        origin_center = grid_centers[origin_idx]
        # 快速获取OD矩阵中的索引
        try:
            orig_od_idx = grid_to_v1[origin_idx]
        except KeyError:
            continue

        for dest_idx in grid_centers:
            if dest_idx == origin_idx or dest_idx not in v1_set:
                continue
            # 快速获取OD矩阵中的索引
            try:
                dest_od_idx = grid_to_v1[dest_idx]
            except KeyError:
                continue

            # 获取OD值
            od_value = od_matrix[orig_od_idx, dest_od_idx]
            if od_value < 1:
                continue

            # 确定颜色类型
            if od_value <= COLOR_CONFIG["light"]["range"][1]:
                color_type = "light"
            elif od_value <= COLOR_CONFIG["middle"]["range"][1]:
                color_type = "middle"
            else:
                color_type = "red"

            # 生成弧线坐标
            dest_center = grid_centers[dest_idx]
            arc_coords = generate_arc_coordinates(origin_center, dest_center)
            # 存储弧线数据
            arc_data[color_type].append({
                "coords": arc_coords,
                "config": COLOR_CONFIG[color_type]
            })

    # 4. 按顺序绘制弧线（先浅色，再中间色，最后红色）
    for color_type in DRAW_ORDER:
        for arc in arc_data[color_type]:
            coords = arc["coords"]
            config = arc["config"]
            ax.plot(
                [p[0] for p in coords],
                [p[1] for p in coords],
                color=config["color"],
                linewidth=config["linewidth"],
                alpha=config["alpha"],
                zorder=config["zorder"]
            )

    # 5. 添加子图标注（位于子图下方中央）
    ax.text(0.5, -0.05, label, transform=ax.transAxes, ha='center', va='top',
            fontsize=14, weight='bold')


# ---------- 主可视化函数 ----------
def plot_od_grid():
    # 1. 初始化2行3列的画布
    fig, axes = plt.subplots(2, 3, figsize=fig_size, dpi=dpi)
    axes = axes.flatten()  # 展平为一维数组，方便遍历

    # 2. 遍历OD数据绘制子图
    for i, (label, file_path) in enumerate(od_sources.items()):
        if i >= len(axes):
            print(f"警告：子图数量({len(axes)})不足，跳过{label}")
            break
        od_matrix = np.load(file_path)[time_step]  # [110, 110]
        plot_od_subplot(axes[i], od_matrix, subplot_labels[i])

    # 3. 隐藏多余的子图（如果有）
    for i in range(len(od_sources), len(axes)):
        axes[i].axis('off')

    # 4. 调整布局并保存
    plt.tight_layout()
    output_file = f"Fig_OD_line_in_map_v2.png"
    plt.savefig(output_file, format='png', bbox_inches='tight', pad_inches=0.1)
    # 可选：保存为PDF
    # plt.savefig("od_all_methods.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ 多方法OD对比图已保存为 {output_file}")


# ---------- 执行绘制 ----------
if __name__ == "__main__":
    plot_od_grid()