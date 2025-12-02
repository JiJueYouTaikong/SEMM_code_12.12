import numpy as np

# 加载原始数据
speed = np.load("../data/Speed_完整批处理_3.17_Final.npy")  # [T,N]
od = np.load("../data/OD_完整批处理_3.17_Final.npy")  # [T,N,N]

# 提取需要的数据段
s77 = speed[:100, 77]
production = np.sum(od, axis=-1)
p77 = production[:100, 77]

# 保存数据到文本文件
with open("s77_data.txt", "w") as file:
    for value in s77:
        file.write(f"{value}\n")

with open("p77_data.txt", "w") as file:
    for value in p77:
        file.write(f"{value}\n")

print("数据已成功保存到 s77_data.txt 和 p77_data.txt")