import pandas as pd
import numpy as np
from datetime import timedelta


def build_time_varying_od_matrix(csv_path, time_interval_min=15):
    """
    从行程CSV文件构建形状为 (T, N, N) 的时变OD矩阵（优化版，支持大规模数据快速处理）
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
    N = len(valid_ids)  # N=70（新增了ID=4，注意此处数量变化）
    # 建立ID与矩阵索引的双向映射（ID->0-69，0-69->ID）
    id_to_index = {id_val: idx for idx, id_val in enumerate(valid_ids)}
    index_to_id = {idx: id_val for idx, id_val in enumerate(valid_ids)}

    # ---------------------- 步骤2：高效读取并预处理CSV文件（核心优化1：按需读列） ----------------------
    # 只读取需要的列，减少内存占用和处理时间（无需读取dropoff时间、passenger_count）
    usecols = ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID']
    df = pd.read_csv(
        csv_path,
        usecols=usecols,  # 按需读取，大幅减少内存
        dtype={  # 提前指定数据类型，避免Pandas自动推断耗时且占用更多内存
            'PULocationID': 'float64',  # 先读为float64，方便后续处理空值
            'DOLocationID': 'float64'
        }
    )

    # 转换出发时间为datetime类型（向量化操作，效率远高于循环）
    df['tpep_pickup_datetime'] = pd.to_datetime(
        df['tpep_pickup_datetime'],
        format='%m/%d/%Y %I:%M:%S %p',
        errors='coerce'  # 忽略格式错误的行（如有）
    )

    # ---------------------- 步骤3：数据清洗 - 仅保留合法ID的行程（向量化操作） ----------------------
    # 过滤空值（向量化操作，比dropna略高效，且同时过滤时间格式错误的行）
    df = df[
        (df['PULocationID'].notna()) &
        (df['DOLocationID'].notna()) &
        (df['tpep_pickup_datetime'].notna())
        ]

    # 转换ID为整数类型（向量化操作）
    df['PULocationID'] = df['PULocationID'].astype(np.int32)
    df['DOLocationID'] = df['DOLocationID'].astype(np.int32)

    # 过滤合法ID（向量化操作，使用isin，效率远高于循环判断）
    valid_ids_set = set(valid_ids)  # 集合查找比列表快，大规模数据更明显
    df_valid = df[
        (df['PULocationID'].isin(valid_ids_set)) &
        (df['DOLocationID'].isin(valid_ids_set))
        ].reset_index(drop=True, inplace=False)  # inplace=False避免修改原数据，同时减少内存碎片

    if len(df_valid) == 0:
        raise ValueError("清洗后无有效行程数据，请检查合法ID或CSV文件内容")

    # ---------------------- 步骤4：时间片划分 - 自动计算T（核心优化2：数值计算生成时间片索引） ----------------------
    # 获取所有出发时间的最小值和最大值
    min_pickup_time = df_valid['tpep_pickup_datetime'].min()
    max_pickup_time = df_valid['tpep_pickup_datetime'].max()

    # 对齐时间片起始点
    align_minute = (min_pickup_time.minute // time_interval_min) * time_interval_min
    min_pickup_aligned = min_pickup_time.replace(minute=align_minute, second=0, microsecond=0)

    # 生成所有时间片的起始时间
    time_slots = []
    current_time = min_pickup_aligned
    while current_time <= max_pickup_time:
        time_slots.append(current_time)
        current_time += timedelta(minutes=time_interval_min)

    T = len(time_slots)
    print(f"时间粒度：{time_interval_min}分钟，时间片总数T：{T}，合法ID数量N：{N}")

    # 核心优化：用时间戳差值直接计算时间片索引（向量化操作，无循环）
    # 1. 计算每个出发时间与对齐后最小时间的时间差（单位：秒）
    time_diff_seconds = (df_valid['tpep_pickup_datetime'] - min_pickup_aligned).dt.total_seconds()
    # 2. 转换为分钟，再除以时间粒度，取整得到时间片索引（向量化，瞬间完成）
    df_valid['time_slot_idx'] = (time_diff_seconds // (time_interval_min * 60)).astype(np.int32)
    # 3. 边界处理：确保索引不超过T-1（极少数超出最大时间片的行）
    df_valid['time_slot_idx'] = df_valid['time_slot_idx'].clip(0, T - 1)

    # ---------------------- 步骤5：构建 (T, N, N) 时变OD矩阵（核心优化3：分组聚合替代iterrows） ----------------------
    # 步骤5.1：将OD ID映射为矩阵索引（向量化操作）
    df_valid['pu_idx'] = df_valid['PULocationID'].map(id_to_index)
    df_valid['do_idx'] = df_valid['DOLocationID'].map(id_to_index)

    # 步骤5.2：分组聚合，统计每个（time_slot_idx, pu_idx, do_idx）的行程数量
    # groupby + size 是Pandas中统计分组数量的高效方式（向量化，无循环）
    od_counts = df_valid.groupby(['time_slot_idx', 'pu_idx', 'do_idx']).size().reset_index(name='count')

    # 步骤5.3：初始化OD矩阵（使用np.int32，减少内存占用）
    od_matrix = np.zeros((T, N, N), dtype=np.int32)

    # 步骤5.4：批量赋值（利用数组索引，向量化操作，远快于逐行赋值）
    # 提取分组后的索引和计数，直接赋值到矩阵中
    t_idxs = od_counts['time_slot_idx'].values
    pu_idxs = od_counts['pu_idx'].values
    do_idxs = od_counts['do_idx'].values
    counts = od_counts['count'].values
    od_matrix[t_idxs, pu_idxs, do_idxs] = counts

    return od_matrix, time_slots, id_to_index, index_to_id


# ---------------------- 可选：超大CSV分块读取函数（数据超过内存时使用） ----------------------
def build_od_matrix_with_chunks(csv_path, time_interval_min=15, chunksize=10 ** 6):
    """
    分块读取超大CSV文件，构建时变OD矩阵（避免内存溢出）
    chunksize：每次读取的行数，根据内存大小调整（默认100万行）
    """
    valid_ids = [
        4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68,
        74, 75, 79, 87, 88, 90, 100, 103, 104, 105,
        107, 113, 114, 116, 120, 125, 127, 128, 137, 140,
        141, 142, 143, 144, 148, 151, 152, 153, 158, 161,
        162, 163, 164, 166, 170, 186, 194, 202, 209, 211,
        224, 229, 230, 231, 232, 233, 234, 236, 237, 238,
        239, 243, 244, 246, 249, 261, 262, 263
    ]
    N = len(valid_ids)
    id_to_index = {id_val: idx for idx, id_val in enumerate(valid_ids)}

    # 第一步：先遍历所有分块，获取完整的时间范围（用于确定T和time_slots）
    min_pickup = None
    max_pickup = None
    usecols = ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID']
    valid_ids_set = set(valid_ids)

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        # 预处理当前分块
        chunk['tpep_pickup_datetime'] = pd.to_datetime(
            chunk['tpep_pickup_datetime'],
            format='%m/%d/%Y %I:%M:%S %p',
            errors='coerce'
        )
        chunk = chunk[
            (chunk['PULocationID'].notna()) &
            (chunk['DOLocationID'].notna()) &
            (chunk['tpep_pickup_datetime'].notna())
            ]
        chunk['PULocationID'] = chunk['PULocationID'].astype(np.int32)
        chunk['DOLocationID'] = chunk['DOLocationID'].astype(np.int32)
        chunk = chunk[
            (chunk['PULocationID'].isin(valid_ids_set)) &
            (chunk['DOLocationID'].isin(valid_ids_set))
            ]

        if len(chunk) == 0:
            continue

        # 更新全局时间范围
        chunk_min = chunk['tpep_pickup_datetime'].min()
        chunk_max = chunk['tpep_pickup_datetime'].max()
        if min_pickup is None or chunk_min < min_pickup:
            min_pickup = chunk_min
        if max_pickup is None or chunk_max > max_pickup:
            max_pickup = chunk_max

    if min_pickup is None or max_pickup is None:
        raise ValueError("清洗后无有效行程数据，请检查合法ID或CSV文件内容")

    # 生成时间片
    align_minute = (min_pickup.minute // time_interval_min) * time_interval_min
    min_pickup_aligned = min_pickup.replace(minute=align_minute, second=0, microsecond=0)
    time_slots = []
    current_time = min_pickup_aligned
    while current_time <= max_pickup:
        time_slots.append(current_time)
        current_time += timedelta(minutes=time_interval_min)
    T = len(time_slots)
    print(f"时间粒度：{time_interval_min}分钟，时间片总数T：{T}，合法ID数量N：{N}")

    # 第二步：分块统计，构建OD矩阵
    od_matrix = np.zeros((T, N, N), dtype=np.int32)
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        # 预处理当前分块（同第一步）
        chunk['tpep_pickup_datetime'] = pd.to_datetime(
            chunk['tpep_pickup_datetime'],
            format='%m/%d/%Y %I:%M:%S %p',
            errors='coerce'
        )
        chunk = chunk[
            (chunk['PULocationID'].notna()) &
            (chunk['DOLocationID'].notna()) &
            (chunk['tpep_pickup_datetime'].notna())
            ]
        chunk['PULocationID'] = chunk['PULocationID'].astype(np.int32)
        chunk['DOLocationID'] = chunk['DOLocationID'].astype(np.int32)
        chunk = chunk[
            (chunk['PULocationID'].isin(valid_ids_set)) &
            (chunk['DOLocationID'].isin(valid_ids_set))
            ]

        if len(chunk) == 0:
            continue

        # 计算当前分块的时间片索引
        time_diff_seconds = (chunk['tpep_pickup_datetime'] - min_pickup_aligned).dt.total_seconds()
        chunk['time_slot_idx'] = (time_diff_seconds // (time_interval_min * 60)).astype(np.int32)
        chunk['time_slot_idx'] = chunk['time_slot_idx'].clip(0, T - 1)

        # 映射OD索引
        chunk['pu_idx'] = chunk['PULocationID'].map(id_to_index)
        chunk['do_idx'] = chunk['DOLocationID'].map(id_to_index)

        # 分组统计当前分块
        chunk_counts = chunk.groupby(['time_slot_idx', 'pu_idx', 'do_idx']).size().reset_index(name='count')

        # 累加到全局OD矩阵
        t_idxs = chunk_counts['time_slot_idx'].values
        pu_idxs = chunk_counts['pu_idx'].values
        do_idxs = chunk_counts['do_idx'].values
        counts = chunk_counts['count'].values
        od_matrix[t_idxs, pu_idxs, do_idxs] += counts

    index_to_id = {idx: id_val for idx, id_val in enumerate(valid_ids)}
    return od_matrix, time_slots, id_to_index, index_to_id


# ---------------------- 调用示例 ----------------------
if __name__ == "__main__":
    # 替换为你的行程CSV文件路径
    start = "0501"
    end =  "0731"
    CSV_FILE_PATH = f"raw_data/2023_Yellow_Taxi_Trip_Data_{start}-{end}.csv"

    # 自定义时间粒度
    TIME_INTERVAL = 5

    # 构建时变OD矩阵（常规大规模数据使用这个）
    try:
        od_mat, time_slots, id2idx, idx2id = build_time_varying_od_matrix(
            csv_path=CSV_FILE_PATH,
            time_interval_min=TIME_INTERVAL
        )

        # 输出矩阵形状并保存
        # 输出矩阵形状（验证是否为 (T, N, N)）
        print(f"\n时变OD矩阵形状：{od_mat.shape}\n", od_mat[0,:20,:20], od_mat.sum())

        np.save(f"../dataset/Manhattan_{TIME_INTERVAL}min_{start}_{end}_T_{od_mat.shape[0]}_Sum_{od_mat.sum()}_最终FINAL.npy", od_mat)

        # 示例查询
        pu_id_example = 90
        do_id_example = 68
        t_idx_example = 0

        pu_idx_example = id2idx[pu_id_example]
        do_idx_example = id2idx[do_id_example]
        trip_count = od_mat[t_idx_example, pu_idx_example, do_idx_example]

        print(f"\n时间片{t_idx_example}（{time_slots[t_idx_example]}）：")
        print(f"出发地ID={pu_id_example} -> 目的地ID={do_id_example}，行程数量：{trip_count}")

    except MemoryError:
        # 若内存溢出，使用分块读取函数
        print("内存不足，切换为分块读取模式...")
        od_mat, time_slots, id2idx, idx2id = build_od_matrix_with_chunks(
            csv_path=CSV_FILE_PATH,
            time_interval_min=TIME_INTERVAL,
            chunksize=10 ** 6
        )
    except Exception as e:
        print(f"程序运行出错：{e}")