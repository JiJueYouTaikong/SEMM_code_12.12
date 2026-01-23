import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def build_time_varying_od_matrix(csv_path, time_interval_min=15):
    """
    从行程CSV文件构建形状为 (T, N, N) 的时变OD矩阵
    参数：
        csv_path: 行程CSV文件路径
        time_interval_min: 时间粒度（分钟），默认15分钟
    返回：
        od_matrix: 时变OD矩阵 (T, N, N)
        time_slots: 时间片列表（每个时间片的起始时间，用于查看T对应的具体时间）
        id_to_index: 合法ID到矩阵索引的映射字典
        index_to_id: 矩阵索引到合法ID的映射字典
    """
    # ---------------------- 步骤1：定义合法ID列表 ----------------------
    valid_ids = [
        4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68,
        74, 75, 79, 87, 88, 90, 100, 103, 104, 105,
        107, 113, 114, 116, 120, 125, 127, 128, 137, 140,
        141, 142, 143, 144, 148, 151, 152, 153, 158, 161,
        162, 163, 164, 166, 170, 186, 194, 202, 209, 211,
        224, 229, 230, 231, 232, 233, 234, 236, 237, 238,
        239, 243, 244, 246, 249, 261, 262, 263
    ]
    N = len(valid_ids)  # N=69
    # 建立ID与矩阵索引的双向映射（ID->0-68，0-68->ID）
    id_to_index = {id_val: idx for idx, id_val in enumerate(valid_ids)}
    index_to_id = {idx: id_val for idx, id_val in enumerate(valid_ids)}

    # ---------------------- 步骤2：读取并预处理CSV文件 ----------------------
    # 读取CSV，注意日期格式（MM/DD/YYYY HH:MM:SS AM/PM）
    df = pd.read_csv(csv_path)

    # 转换出发/到达时间为datetime类型
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

    # 处理可能的空值（PULocationID/DOLocationID为空的行程直接过滤）
    df = df.dropna(subset=['PULocationID', 'DOLocationID'])

    # 转换ID为整数类型（避免后续匹配问题）
    df['PULocationID'] = df['PULocationID'].astype(int)
    df['DOLocationID'] = df['DOLocationID'].astype(int)

    # ---------------------- 步骤3：数据清洗 - 仅保留合法ID的行程 ----------------------
    df_valid = df[
        (df['PULocationID'].isin(valid_ids)) &
        (df['DOLocationID'].isin(valid_ids))
        ].reset_index(drop=True)

    if len(df_valid) == 0:
        raise ValueError("清洗后无有效行程数据，请检查合法ID或CSV文件内容")

    # ---------------------- 步骤4：时间片划分 - 自动计算T ----------------------
    # 获取所有出发时间的最小值和最大值（确定时间范围）
    min_pickup_time = df_valid['tpep_pickup_datetime'].min()
    max_pickup_time = df_valid['tpep_pickup_datetime'].max()

    # 对齐时间片起始点（例如：将min_pickup_time对齐到最近的time_interval_min倍数）
    # 示例：8:45:00 若粒度15分钟，对齐后仍为8:45:00；若粒度30分钟，对齐为8:30:00
    align_minute = (min_pickup_time.minute // time_interval_min) * time_interval_min
    min_pickup_aligned = min_pickup_time.replace(minute=align_minute, second=0, microsecond=0)

    # 生成所有时间片的起始时间（直到覆盖max_pickup_time）
    time_slots = []
    current_time = min_pickup_aligned
    while current_time <= max_pickup_time:
        time_slots.append(current_time)
        current_time += timedelta(minutes=time_interval_min)

    T = len(time_slots)  # 自动计算时间片总数T
    print(f"时间粒度：{time_interval_min}分钟，时间片总数T：{T}，合法ID数量N：{N}")

    # 为每个有效行程分配对应的时间片索引（即属于第几个T）
    def get_time_slot_index(pickup_time):
        for idx, slot_start in enumerate(time_slots):
            slot_end = slot_start + timedelta(minutes=time_interval_min)
            if slot_start <= pickup_time < slot_end:
                return idx
        # 若超过最后一个时间片，分配到最后一个（边界处理）
        return T - 1

    df_valid['time_slot_idx'] = df_valid['tpep_pickup_datetime'].apply(get_time_slot_index)

    # ---------------------- 步骤5：构建 (T, N, N) 时变OD矩阵 ----------------------
    # 初始化OD矩阵（全0，形状(T, N, N)）
    od_matrix = np.zeros((T, N, N), dtype=np.int32)

    # 遍历所有有效行程，统计OD对数量
    for _, row in df_valid.iterrows():
        t_idx = row['time_slot_idx']  # 时间片索引（T维度）
        pu_id = row['PULocationID']  # 出发地ID
        do_id = row['DOLocationID']  # 目的地ID

        # 映射为矩阵的行索引（出发地）和列索引（目的地）
        pu_idx = id_to_index[pu_id]
        do_idx = id_to_index[do_id]

        # 对应位置计数+1
        od_matrix[t_idx, pu_idx, do_idx] += 1

    return od_matrix, time_slots, id_to_index, index_to_id


# ---------------------- 调用示例 ----------------------
if __name__ == "__main__":
    # 替换为你的行程CSV文件路径
    CSV_FILE_PATH = "raw_data/2023_Yellow_Taxi_Trip_Data_0501-0501.csv"

    # 自定义时间粒度（这里用15分钟，可改为30、60等）
    TIME_INTERVAL = 30

    # 构建时变OD矩阵
    try:
        od_mat, time_slots, id2idx, idx2id = build_time_varying_od_matrix(
            csv_path=CSV_FILE_PATH,
            time_interval_min=TIME_INTERVAL
        )

        # 输出矩阵形状（验证是否为 (T, N, N)）
        print(f"\n时变OD矩阵形状：{od_mat.shape}\n",od_mat[0],od_mat.sum())

        # np.save(f"../dataset/Manhattan_{TIME_INTERVAL}min", od_mat)

        # 示例：查看第0个时间片、出发地ID=90、目的地ID=68的行程数量
        pu_id_example = 90
        do_id_example = 68
        t_idx_example = 0

        pu_idx_example = id2idx[pu_id_example]
        do_idx_example = id2idx[do_id_example]
        trip_count = od_mat[t_idx_example, pu_idx_example, do_idx_example]

        print(f"\n时间片{t_idx_example}（{time_slots[t_idx_example]}）：")
        print(f"出发地ID={pu_id_example} -> 目的地ID={do_id_example}，行程数量：{trip_count}")

    except Exception as e:
        print(f"程序运行出错：{e}")