import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子


# 加载数据
def load_data():
    set_seed(42)

    # 加载数据
    speed = np.load('data/Link_Speed_25.2.25最新TL.npy')  # [T,L]
    flow = np.load('data/Link_Flow_25.2.25最新TL.npy')    # [T,L]

    od = np.load('../data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]
    # od = np.load('../data/OD_完整批处理_3.17_Final.npy')  # 形状 [T, N, N]

    # 获取数据长度
    T, _, _ = od.shape

    # 测试OD中的始终全零值
    # od_mean = od.sum(axis=0)  # [N, N]
    # non_zero_indices = np.nonzero(od_mean)
    # print("-----------------------------------")
    # print(od_mean[:10, :10])
    # print(non_zero_indices)
    #
    # # 计算非零元素的数量，即 num_of_links
    # num_of_links = len(non_zero_indices[0])
    # print(num_of_links)
    # print("-----------------------------------")


    # 计算OD的均值和标准差
    mean_all = np.mean(od)
    std_all = np.std(od)

    print("所有元素的均值:", mean_all)
    print("所有元素的标准差:", std_all)

    od = od.reshape(T,-1)

    N = od.shape[-1]



    L = speed.shape[-1]
    print(f"T:{T}, N:{N}, L:{L}")

    # # 设置均值和标准差
    # mean = 0.05
    # std_dev = 0.01
    # # 获取数据长度 T
    #
    # # 生成T组高斯分布数据
    # x_random = np.random.normal(loc=mean, scale=std_dev, size=(T, N))
    #
    # # 将数据裁剪到0到1之间
    # x_random = np.clip(x_random, 0, 0.1)

    # 按顺序划分数据
    train_size = int(T * 0.6)
    val_size = int(T * 0.2)

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    flow_train, flow_val, flow_test = flow[train_indices], flow[val_indices], flow[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    # x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # 输出划分后的数据形状
    print("6:2:2顺序划分的训练集", speed_train.shape,flow_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集", speed_test.shape, "OD", od_test.shape)



    speed_train = torch.tensor(speed_train, dtype=torch.float32)
    speed_val = torch.tensor(speed_val, dtype=torch.float32)
    speed_test = torch.tensor(speed_test, dtype=torch.float32)

    flow_train = torch.tensor(flow_train, dtype=torch.float32)
    flow_val = torch.tensor(flow_val, dtype=torch.float32)
    flow_test = torch.tensor(flow_test, dtype=torch.float32)

    od_train = torch.tensor(od_train, dtype=torch.float32)
    od_val = torch.tensor(od_val, dtype=torch.float32)
    od_test = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("训练集shape:", flow_train.shape, speed_train.shape, od_train.shape)

    T_test = od_test.shape[0]
    T_val = od_val.shape[0]

    # 1. 从OD统计的高斯噪声采样相似度极差
    noise_dim = 50
    z = np.zeros((T_test, noise_dim))

    # 进行 T 次噪声采样
    for i in range(T_test):
        # 从高斯分布（均值为 0，标准差为 1）中采样 N 个值
        z[i] = np.random.normal(loc=mean_all, scale=std_all, size=noise_dim)

    z=torch.tensor(z, dtype=torch.float32)
    print("z ",z.shape)

    noise_dim = 50
    z_val = np.zeros((T_val, noise_dim))

    # 进行 T 次噪声采样
    for i in range(T_val):
        # 从高斯分布（均值为 0，标准差为 1）中采样 N 个值
        z_val[i] = np.random.normal(loc=mean_all, scale=std_all, size=noise_dim)

    z_val = torch.tensor(z_val, dtype=torch.float32)
    print("z_val ", z_val.shape)

    # 2.修改为蒙特卡洛方法
    # z = np.load("data/OVS的噪声z.npy")
    # z = z.reshape(T_test,-1)
    # z=torch.tensor(z, dtype=torch.float32)
    # print("z ", z.shape)
    #
    #
    # z_val = np.load("data/OVS的噪声z_val.npy")
    # print("z val:",z_val.shape)
    # T_val, n, n = z_val.shape
    # z_val = z_val.reshape(T_val, n*n)
    # z_val = torch.tensor(z_val, dtype=torch.float32)
    # print("z_val ", z_val.shape)

    train_dataset = TensorDataset(od_train, flow_train, speed_train)
    val_dataset = TensorDataset(z_val, od_val, flow_val, speed_val)
    test_dataset = TensorDataset(z, od_test, flow_test, speed_test)



    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader,T,N,L