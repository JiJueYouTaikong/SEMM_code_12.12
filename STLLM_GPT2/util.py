import numpy as np
import os
import scipy.sparse as sp
import torch
import pickle


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
def load_data(data):

    # 加载数据
    # data = np.load('data/武汉速度数据集_1KM_110区域_25.1.14.npy')  # 形状 [T, N, 2]
    # speed = data[:, :, 0]  # 平均速度 [T, N]
    #
    # od = np.load('data/武汉OD数据集_1KM_110区域_过滤cnt_对角线0_25.1.14.npy')  # 形状 [T, N, N]
    if "Speed" in data:
        speed = np.load('data/Speed_完整批处理_3.17_Final.npy')
        od = np.load('data/OD_完整批处理_3.17_Final.npy')

        od_production = np.sum(od, axis=-1)  # 形状 [T_train, N]

        # 获取数据长度 T
        T, N = speed.shape

        # 按顺序划分数据
        train_size = int(T * 0.6)
        val_size = int(T * 0.2)
    elif "MCM" in data:
        speed = np.load('data/Speed_完整批处理_3.17_Final_MCM_60.npy')
        od = np.load('data/OD_完整批处理_3.17_Final_MCM_60.npy')

        od_production = np.sum(od, axis=-1)  # 形状 [T_train, N]

        # 获取数据长度 T
        T, N = speed.shape

        # 按顺序划分数据
        train_size = int(T * 0.9537)
        val_size = int(T * 0.0225)

    # 顺序划分索引
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, T)

    # 按索引划分数据
    speed_train, speed_val, speed_test = speed[train_indices], speed[val_indices], speed[test_indices]
    od_train, od_val, od_test = od[train_indices], od[val_indices], od[test_indices]
    # x_random_train, x_random_val, x_random_test = x_random[train_indices], x_random[val_indices], x_random[test_indices]

    # 输出划分后的数据形状
    print("6:2:2顺序划分的训练集Speed", speed_train.shape, "OD", od_train.shape)
    print("6:2:2顺序划分的验证集Speed", speed_val.shape, "OD", od_val.shape)
    print("6:2:2顺序划分的测试集Speed", speed_test.shape, "OD", od_test.shape)

    # 在训练集上计算 OD 出发总量
    od_train_departures = np.sum(od_train, axis=-1)  # 形状 [T_train, N]



    # 计算每个区域在 T_train 时间步的平均速度和平均出发总量
    mean_speed = np.mean(speed_train, axis=0)  # 每个区域的平均速度 [N,]
    mean_departures = np.mean(od_train_departures, axis=0)  # 每个区域的平均出发总量 [N,]
    # 计算平均速度和平均出发总量的差值
    temporal = (mean_departures - mean_speed).astype(float)  # [N,]

    # freq
    # speed_freq = np.load('data/速度的周期状态_对应25.1.14的速度数据集.npy')  # 形状 [N,]
    # od_freq = np.load('data/OD的周期状态_对应25.1.14的OD数据集.npy')  # 形状 [N,]
    #
    # freq = (od_freq - speed_freq).astype(float)

    # speed_freq = get_X_Freq(speed)
    # od_freq = get_X_Freq(od_production)
    # freq = (od_freq - speed_freq).astype(float)

    # 归一化
    scaler = MinMaxScaler()
    # print(f"test_data: {speed_test[-2, 10:20]}")
    train_data = scaler.fit_transform(speed_train.reshape(-1, 1)).reshape(speed_train.shape)
    val_data = scaler.transform(speed_val.reshape(-1, 1)).reshape(speed_val.shape)
    test_data = scaler.transform(speed_test.reshape(-1, 1)).reshape(speed_test.shape)

    # print(f"train_data: {train_data[:10, 66]}")

    # print(f"test_data: {test_data[-2, 10:20]}")

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_target = torch.tensor(od_train, dtype=torch.float32)
    val_target = torch.tensor(od_val, dtype=torch.float32)
    test_target = torch.tensor(od_test, dtype=torch.float32)

    # 打印结果形状
    print("归一化后的训练集 shape:", train_data.shape, "OD形状", train_target.shape)
    print("归一化后的验证集 shape:", val_data.shape, "OD形状", val_target.shape)
    print("归一化后的测试集 shape:", test_data.shape, "OD形状", test_target.shape)

    train_dataset = TensorDataset(train_data, train_target)
    val_dataset = TensorDataset(val_data, val_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # scaler = MinMaxScaler()
    # x_temp = scaler.fit_transform(temporal.reshape(-1, 1))
    # scaler = MinMaxScaler()
    # x_freq = scaler.fit_transform(freq.reshape(-1, 1))
    # x_temp = np.squeeze(x_temp, axis=-1)
    # x_freq = np.squeeze(x_freq, axis=-1)

    return train_loader, val_loader, test_loader,scaler



# class DataLoader(object):
#     def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
#         self.batch_size = batch_size
#         self.current_ind = 0
#         if pad_with_last_sample:
#             num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
#             x_padding = np.repeat(xs[-1:], num_padding, axis=0)
#             y_padding = np.repeat(ys[-1:], num_padding, axis=0)
#             xs = np.concatenate([xs, x_padding], axis=0)
#             ys = np.concatenate([ys, y_padding], axis=0)
#         self.size = len(xs)
#         self.num_batch = int(self.size // self.batch_size)
#         self.xs = xs
#         self.ys = ys
#
#     def shuffle(self):
#         permutation = np.random.permutation(self.size)
#         xs, ys = self.xs[permutation], self.ys[permutation]
#         self.xs = xs
#         self.ys = ys
#
#     def get_iterator(self):
#         self.current_ind = 0
#
#         def _wrapper():
#             while self.current_ind < self.num_batch:
#                 start_ind = self.batch_size * self.current_ind
#                 end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
#                 x_i = self.xs[start_ind:end_ind, ...]
#                 y_i = self.ys[start_ind:end_ind, ...]
#                 yield (x_i, y_i)
#                 self.current_ind += 1
#
#         return _wrapper()
#
#
# class StandardScaler:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def transform(self, data):
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean
#
#
# # def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
# #     data = {}
# #     for category in ["train", "val", "test"]:
# #         cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
# #         data["x_" + category] = cat_data["x"]
# #         data["y_" + category] = cat_data["y"]
# #     scaler = StandardScaler(
# #         mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
# #     )
# #     # Data format
# #     for category in ["train", "val", "test"]:
# #         data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
# #
# #     print("Perform shuffle on the dataset")
# #     random_train = torch.arange(int(data["x_train"].shape[0]))
# #     random_train = torch.randperm(random_train.size(0))
# #     data["x_train"] = data["x_train"][random_train, ...]
# #     data["y_train"] = data["y_train"][random_train, ...]
# #
# #     random_val = torch.arange(int(data["x_val"].shape[0]))
# #     random_val = torch.randperm(random_val.size(0))
# #     data["x_val"] = data["x_val"][random_val, ...]
# #     data["y_val"] = data["y_val"][random_val, ...]
# #
# #     # random_test = torch.arange(int(data['x_test'].shape[0]))
# #     # random_test = torch.randperm(random_test.size(0))
# #     # data['x_test'] =  data['x_test'][random_test,...]
# #     # data['y_test'] =  data['y_test'][random_test,...]
# #
# #     data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
# #     data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
# #     data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
# #     data["scaler"] = scaler
# #
# #     return data
# def load_dataset(dataset_dirs, batch_size, valid_batch_size=None, test_batch_size=None):
#     data = {}
#     # 遍历多个数据集目录
#     for dataset_dir in dataset_dirs:
#         for category in ["train", "val", "test"]:
#             file_path = os.path.join('data', dataset_dir, category + ".npz")  # 添加 'data' 作为前缀
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"File not found: {file_path}")
#             cat_data = np.load(file_path)
#             if category == "train":
#                 data["x_train"] = cat_data["x"]
#                 data["y_train"] = cat_data["y"]
#             elif category == "val":
#                 data["x_val"] = cat_data["x"]
#                 data["y_val"] = cat_data["y"]
#             elif category == "test":
#                 data["x_test"] = cat_data["x"]
#                 data["y_test"] = cat_data["y"]
#
#     # 继续其他的处理逻辑
#     scaler = StandardScaler(
#         mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
#     )
#     for category in ["train", "val", "test"]:
#         data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
#
#     # 打乱数据并创建数据加载器
#     random_train = torch.randperm(data["x_train"].shape[0])
#     data["x_train"] = data["x_train"][random_train, ...]
#     data["y_train"] = data["y_train"][random_train, ...]
#
#     random_val = torch.randperm(data["x_val"].shape[0])
#     data["x_val"] = data["x_val"][random_val, ...]
#     data["y_val"] = data["y_val"][random_val, ...]
#
#     data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
#     data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
#     data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
#     data["scaler"] = scaler
#
#     return data



def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, real):
    mae = MAE_torch(pred, real, None).item()
    mape = MAPE_torch(pred, real,0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, None).item()
    return mae, mape, rmse, wmape


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise e
    return pickle_data

