import numpy as np


def convert_flow(flow, adj):
    # 找出邻接矩阵中所有非零元素的位置
    links = np.nonzero(adj)
    # 计算链路数量
    L = len(links[0])
    T = flow.shape[0]
    # 初始化转换后的流量矩阵
    flow_L = np.zeros((T, L))
    # 对于每个时间步
    for t in range(T):
        # 从flow[t]中提取与链路对应的流量值
        flow_L[t] = flow[t][links]
    return flow_L



flow = np.load("../data/Link_flow_TNN_MSA-SUE_logit3_6_14.npy")
adj = np.load("../data/adj110_3.17.npy")

# 转换流量数据
flow_L = convert_flow(flow, adj)
print("转换后的流量数据形状:", flow_L.shape)
np.save("../data/Link_flow_TL_MSA-SUE_logit3_6_14.npy",flow_L)