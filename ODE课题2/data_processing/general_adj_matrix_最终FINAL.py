# 导入所需库
import pandas as pd
import numpy as np


def csv_adj_to_npy(csv_path, npy_save_path):
    """
    将带表头的邻接矩阵CSV转换为NPY文件
    :param csv_path: 输入的CSV文件路径
    :param npy_save_path: 输出的NPY文件保存路径
    """
    try:
        # 1. 读取CSV文件，保留表头（默认第一行为表头，第一列为索引列）
        # header=0 表示第一行为列名，index_col=0 表示第一列为行索引（对应节点名称）
        adj_df = pd.read_csv(csv_path, header=0)

        # 2. 验证数据形状（可选，用于排查数据异常）
        print(f"读取到的CSV数据形状：{adj_df.shape}")
        if adj_df.shape != (69, 69):
            # 原始CSV是70行69列，读取后（表头+索引列）应转为69×69的DataFrame
            print("警告：数据形状不符合预期（69×69），将自动截取/补全为69×69矩阵")
            # 截取前69行69列，确保对应69个节点
            adj_df = adj_df.iloc[:69, :69]

        # 3. 将DataFrame转换为numpy数组（邻接矩阵的数值格式）
        adj_matrix = adj_df.values.astype(np.float64)  # 转为浮点型，适配邻接矩阵常见格式（可根据需求改为int）

        # 4. 保存为npy文件
        np.save(npy_save_path, adj_matrix)
        print(f"邻接矩阵已成功保存为npy文件，路径：{npy_save_path}")
        print(f"保存的npy数组形状：{adj_matrix.shape}\n", adj_matrix[:10,:10])

    except FileNotFoundError:
        print(f"错误：未找到文件 {csv_path}，请检查文件路径是否正确")
    except Exception as e:
        print(f"错误：处理过程中出现异常 - {str(e)}")


# 主程序调用
if __name__ == "__main__":
    # 配置文件路径（请确保曼哈顿_adj_matrix.csv和该脚本在同一目录下，否则填写完整绝对路径）
    CSV_FILE = "raw_data/adj_matrix/曼哈顿_adj_matrix.csv"
    NPY_FILE = "../dataset/adj_matrix/曼哈顿_adj_matrix_TNN.npy"

    # 执行转换
    csv_adj_to_npy(CSV_FILE, NPY_FILE)