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
        'weight': 3,
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
    "Pred_ours": "../测试集TNN/Pred_RED-5.7127.npy",
    "SSM": "../测试集TNN/Pred_SSM_SUE_MSA_6.15_7.80.npy",

    "GPT2": "../测试集TNN/Pred_GPT2.npy",
    "DeepGravity": "../测试集TNN/Pred_DeepGravity_6.88.npy",
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
    m = folium.Map(location=[30.4341, 114.5113], zoom_start=13,
                   zoom_control=False,
                   scrollWheelZoom=False,
                   control_scale=False,
                   attribution_control=False
                   )

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
                        weight=3.0,
                        fill=True,
                        fill_color=colors[idx],
                        fill_opacity=1
                    ).add_to(m)
            else:  # Polygon
                folium.Polygon(
                    locations=coords,
                    color='black',
                    weight=3.0,
                    fill=True,
                    fill_color=colors[idx],
                    fill_opacity=1
                ).add_to(m)

    # 添加可拖动图例
    # add_draggable_legend(m)

    m.save(output_file)
    print(f"✅ {title} 地图已保存为 {output_file}")


# 添加可拖动图例
def add_draggable_legend(map_obj):
    # 定义可拖动图例的HTML代码
    legend_html = '''
    <div id="legend" style="position: absolute; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 15px; border: 1px solid #ccc; 
                border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.2); width: 200px;">
        <div id="legend-header" style="cursor: move; padding-bottom: 10px; border-bottom: 1px solid #eee; 
                     margin-bottom: 10px; font-weight: bold; font-size: 16px;">图例</div>
        <div class="legend-item" style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 30px; height: 30px; background-color: #fdcebb; margin-right: 10px;"></div>
            <span style="font-size: 14px;">1-5</span>
        </div>
        <div class="legend-item" style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 30px; height: 30px; background-color: #fcb89e; margin-right: 10px;"></div>
            <span style="font-size: 14px;">5-10</span>
        </div>
        <div class="legend-item" style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 30px; height: 30px; background-color: #fc9373; margin-right: 10px;"></div>
            <span style="font-size: 14px;">10-50</span>
        </div>
        <div class="legend-item" style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 30px; height: 30px; background-color: #db2824; margin-right: 10px;"></div>
            <span style="font-size: 14px;">50-100</span>
        </div>
        <div class="legend-item" style="display: flex; align-items: center;">
            <div style="width: 30px; height: 30px; background-color: #67000d; margin-right: 10px;"></div>
            <span style="font-size: 14px;">100+</span>
        </div>
    </div>

    <script>
    // 使图例可拖动
    document.addEventListener('DOMContentLoaded', function() {
        const legend = document.getElementById('legend');
        const header = document.getElementById('legend-header');
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        let isDragging = false;

        // 只在标题栏上监听鼠标按下事件
        header.onmousedown = dragMouseDown;

        // 鼠标按下时开始拖动
        function dragMouseDown(e) {
            e.preventDefault();
            isDragging = true;
            // 获取鼠标位置
            pos3 = e.clientX;
            pos4 = e.clientY;

            // 添加拖动样式
            legend.style.cursor = 'grabbing';
            legend.style.opacity = '0.9';

            // 鼠标移动时调用dragMouseMove函数
            document.onmousemove = dragMouseMove;

            // 鼠标释放时停止拖动
            document.onmouseup = closeDragElement;
        }

        function dragMouseMove(e) {
            if (!isDragging) return;
            e.preventDefault();

            // 计算新位置
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;

            // 获取地图容器的边界
            const mapContainer = document.querySelector('.leaflet-container');
            const mapRect = mapContainer.getBoundingClientRect();

            // 计算图例的新位置（使用top/left而非bottom/left）
            const newTop = legend.offsetTop - pos2;
            const newLeft = legend.offsetLeft - pos1;

            // 确保图例不会被拖出地图容器
            const legendRect = legend.getBoundingClientRect();
            const maxTop = mapRect.height - legendRect.height;
            const maxLeft = mapRect.width - legendRect.width;

            // 限制在地图容器内
            const boundedTop = Math.max(0, Math.min(newTop, maxTop));
            const boundedLeft = Math.max(0, Math.min(newLeft, maxLeft));

            // 设置新位置
            legend.style.top = boundedTop + "px";
            legend.style.left = boundedLeft + "px";
            legend.style.bottom = "auto"; // 清除bottom属性，使用top定位
        }

        function closeDragElement() {
            // 停止移动
            isDragging = false;
            document.onmouseup = null;
            document.onmousemove = null;

            // 恢复样式
            legend.style.cursor = 'move';
            legend.style.opacity = '1';
        }
    });
    </script>
    '''

    # 将可拖动图例添加到地图上
    # map_obj.get_root().html.add_child(folium.Element(legend_html))


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