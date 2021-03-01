#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 15:10
# @Author  : Cestbo
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import torch
import numpy as np
import matplotlib.pyplot as plt


def smoothness(res):
    """
        度量平滑程度
        :param res:（b, t, n, 1）
        :return:
        """
    b, t, n, _ = res.shape
    res = res.view(b * t, n)
    all_smooth = 0
    for i in range(n):
        one_smooth = 0
        for j in range(n):
            if i != j:
                a = res[:, i] / torch.norm(res[:, i])
                b = res[:, j] / torch.norm(res[:, j])
                one_smooth += torch.norm(a - b).item() / 2
        all_smooth += one_smooth / (n - 1)
    return all_smooth / n

def init_net(net):
    for params in net.parameters(recurse=True):
        if params.dim() > 1:
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        else:
            torch.nn.init.uniform_(params)


def visualization(**dict):
    """
    :param dict: save train_loss and val_loss
    :return:
    """
    train_loss = dict["train_loss"]
    val_loss = dict["val_loss"]
    plt.plot(train_loss, label="training loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()
    plt.show()


def rmse(a, b) -> float:
    return (a - b).pow(2).mean().sqrt().item()


def mae(a, b) -> float:
    return (a - b).abs().mean().item()


def get_adj(adj_path, num_node):
    """

    :param adj_path: 邻接矩阵的路径
    :param num_node: 节点个数
    :return: 归一化后的邻接矩阵
    """
    A = np.eye(num_node)
    for ln in open(adj_path, 'r').readlines()[1:]:
        i, j, d = ln.split(',')
        i, j = int(i), int(j)
        A[i, j] = A[j, i] = 1
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return torch.from_numpy(A_wave).float()


def get_data(data_path, num_his, num_pred):
    """
    生成可训练数据
    :param data_path: 数据路径
    :param num_his: 历史时间步
    :param num_pred: 预测时间步
    :return: （数据，标签，时间嵌入信息）
    """
    data = torch.from_numpy(np.load(data_path)['data'])
    size, num_node, num_feature = data.shape
    num_sample = size - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, num_node, num_feature)
    y = torch.zeros(num_sample, num_pred, num_node, 1)
    e = torch.zeros(num_sample, num_pred, 2)  # 时间特征，每周时间和每天时间。
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred, :, 0:1]  # 只取流量特征
        # 时间特征e
        for j in range(num_pred):
            """
            12个时间步一小时，24小时一天，7天一周
            """
            e[i][j] = torch.tensor([((i + num_his + j) / (12 * 24)) % 7, ((i + num_his + j) / 12) % 24],
                                   dtype=torch.float)
    return x, y, e


def normalization(x, y, e, split1, split2):
    """
    划分数据集并归一化
    :param x: (n, t, n, f)
    :param y: (n, t, n, 1)
    :param e: (n, t, 2)
    :param split1: 划分界限
    :param split2:
    :return: 归一化好后的数据集
    """
    train_x, train_y, train_e = x[:split1], y[:split1], e[:split1]
    val_x, val_y, val_e = x[split1:split2], y[split1:split2], e[split1:split2]
    test_x, test_y, test_e = x[split2:], y[split2:], e[split2:]
    mean, std = torch.mean(train_x), torch.std(train_x)
    train_x, val_x, test_x = (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std
    mean_e, std_e = torch.mean(train_e), torch.std(train_e)
    train_e, val_e, test_e = (train_e - mean_e) / std_e, (val_e - mean_e) / std_e, (test_e - mean_e) / std_e
    return train_x, train_y, val_x, val_y, test_x, test_y, train_e, val_e, test_e, mean, std


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['lines.linewidth'] = 1.2
    data = np.load("data/PeMSD4/PeMSD4.npz")['data']
    a = data[:, 100, 0]
    fig1 = plt.figure(num='fig111111', figsize=(10, 6), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    y = data[:12, :, 0]
    x = [i for i in range(12)]
    plt.plot(x, y[:, 0], linestyle='dashed', label='点0')
    plt.plot(x, y[:, 1], linestyle='dashed', label='点1')
    plt.plot(x, y[:, 2], linestyle='dashed', label='点2')
    plt.plot(x, y[:, 3], linestyle='dashed', label='点3')
    plt.plot(x, y[:, 4], linestyle='dashed', label='点4')
    plt.plot(x, y[:, 5], linestyle='dashed', label='点5')
    plt.plot(x, y[:, 6], linestyle='dashed', label='点6')
    plt.plot(x, y[:, 7], linestyle='dashed', label='点7')
    plt.plot(x, y[:, 8], linestyle='dashed', label='点8')
    # plt.plot(data[12*24,:,0])
    plt.legend()
    plt.xlabel("时间/天")
    plt.ylabel("流量")
    plt.show()
