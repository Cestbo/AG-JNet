#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 10:25
# @Author  : Cestbo
# @Site    : 
# @File    : model.py
# @Software: PyCharm

import torch
import torch.nn as nn


class GCLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super(GCLayer, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_f, out_f))

    def forward(self, x, adj):
        """
        GCN
        :param x:(b, t, n, in_f)
        :param adj: (n , n)
        :return: (b, t, b, out_f)
        """
        x = adj @ x @ self.w
        return x.relu()


#  Adaptive Graph Generation Convolution
class AGGCLayer(nn.Module):
    def __init__(self, in_f, out_f, device, n, d=10):
        """
        AGGC
        :param in_f:
        :param out_f:
        :param device:
        :param n: 节点个数
        :param d: 降维
        """
        super(AGGCLayer, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_f, out_f))
        self.adj_w = nn.Parameter(torch.zeros(n, d))
        self.b = nn.Parameter(torch.zeros(out_f))
        self.device = device
        self.n = n

    def forward(self, x, adj):
        """
        AGGC
        :param x: (b, t, n, in_f)
        :param adj: 不需要
        :return: (b, t, n, out_f)
        """
        # 通过adj_w自适应学习邻接矩阵
        adj = torch.softmax(torch.relu(self.adj_w @ torch.t(self.adj_w)), dim=1)
        I = torch.eye(self.n).to(self.device)
        x = (adj + I) @ x @ self.w + self.b
        return x.relu()


class SM(nn.Module):
    def __init__(self, in_f, out_f, num_layer, device, n):
        """
        Spatial Modeling
        :param in_f:
        :param out_f:
        :param num_layer: 堆叠层数
        :param device:
        :param n: 节点个数
        """
        super(SM, self).__init__()
        self.layers = nn.ModuleList([AGGCLayer(in_f, out_f, device, n) for _ in range(num_layer)])
        # self.layers = nn.ModuleList([GCLayer(in_f, out_f) for _ in range(num_layer)])
        self.w = nn.Parameter(torch.zeros(num_layer * out_f, out_f))

    def forward(self, x, adj):
        """
        Spatial Modeling
        :param x: (b, t, n, f)
        :param adj: (n, n)
        :return:(b, t, n, f)
        """
        y = []
        for layer in self.layers:
            x = layer(x, adj)
            y.append(x)
        y = torch.stack(y, dim=-1)
        y = torch.max(y, dim=-1)[0]  # 测试max效果好
        # y = torch.cat(y, dim=-1)
        # y = y @ self.w
        return y.relu()


class TM(nn.Module):
    def __init__(self, in_f, out_f, num_layers, d=2, k=3):
        """
        Temporal Modeling
        :param in_f:
        :param out_f:
        :param num_layers: 堆叠层数
        :param d: 扩张率
        :param k: 卷积核
        """
        super(TM, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(in_f, out_f, (1, k), padding=(0, d), dilation=(1, d)) for i in
             range(num_layers)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(out_f) for _ in range(num_layers)]
        )
        self.conv = nn.Conv2d(num_layers * out_f, out_f, 1)
        self.bn = nn.BatchNorm2d(out_f)

    def forward(self, x):
        """
        Temporal Modeling
        :param x: (b, f, n, t)
        :return:(b, f, n, t)
        """

        y = []
        for i, layer in enumerate(self.layers):
            bn = self.bns[i]
            x = layer(x)
            x = bn(x)
            x = torch.relu(x)
            y.append(x)
        y = torch.cat(y, dim=1)  # (b, num_layer*f, n, t)
        y = self.bn(self.conv(y))  # (b, out_f, n, t)

        return y.relu()


class FA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(FA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Feature Attention
        :param x: (b, f, n, t)
        :return:  (b, f, n, t)
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TimeE(nn.Module):
    def __init__(self, in_f=2, out_f=6):
        super(TimeE, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.ReLU(),
            nn.Linear(out_f, out_f),
            nn.ReLU(),
            nn.Linear(out_f, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        :param x: (b, t, 2)
        :return: (b, t, 1)
        """
        return self.linear(x)


class STBLock(nn.Module):
    def __init__(self, in_f, out_f, num_spa, num_time, n, num_his=12, num_pred=6, device="cpu"):
        """
        :param in_f:
        :param out_f:
        :param num_spa: 空间建模网络层数
        :param num_time: 时间建模网络层数
        :param n: 节点个数
        :param num_his: 历史时间步
        :param num_pred: 预测时间步
        :param device:
        """
        super(STBLock, self).__init__()
        self.fa = FA(out_f)
        self.spa_block = SM(in_f, out_f, num_spa, device, n)
        self.time_block = TM(in_f, out_f, num_time)
        self.w1 = nn.Parameter(torch.zeros(out_f * 2, out_f))
        # 时间维度转换
        self.time_conv = nn.Sequential(
            nn.Conv2d(out_f, out_f, (1, num_his + 1 - num_pred)),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
        )
        self.time_e = TimeE()

    def forward(self, x, adj, e):
        """

        :param x: (b, t, n, f)
        :param adj: (n, n)
        :param e:时间特征嵌入 (b, t, 2)
        :return:(b, num_pred, n, f)
        """
        x = x.permute(0, 3, 2, 1)  # (b,f,n,t)
        x = self.fa(x)
        spa_output = self.spa_block(x.permute(0, 3, 2, 1), adj)  # (b,t,n,f)
        time_output = self.time_block(x).permute(0, 3, 2, 1)  # (b,t,n,f)
        z = torch.sigmoid(torch.matmul(torch.cat((spa_output, time_output), dim=-1), self.w1))
        y = z * spa_output + (1 - z) * time_output  # (b,t,n,f)
        y = self.time_conv(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # (b,num_pred,n,f)
        time_e = self.time_e(e)  # (b, num_pred, 1)
        y = torch.unsqueeze(time_e, dim=-1) * y  # (b,num_pred,n,f)
        # y = y.permute(0, 2, 1, 3)
        return y.relu()


class AG_JNet(nn.Module):
    def __init__(self, in_f, out_f, num_spa, num_time, n, num_his=12, num_pred=6, device="cpu"):
        """
        Adaptive graph generation jump network
        :param in_f:
        :param out_f:
        :param num_spa: 空间建模网络层数
        :param num_time: 时间建模网络层数
        :param n: 节点个数
        :param num_his: 历史时间步
        :param num_pred: 预测时间步
        :param device:
        """
        super(AG_JNet, self).__init__()
        self.stblock1 = STBLock(out_f, out_f, num_spa, num_time, n, num_his, num_pred, device)
        self.stblock2 = STBLock(out_f, out_f, num_spa, num_time, n, num_pred, num_pred, device)
        # 扩充特征维度
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_f, out_f // 8, 1),
            nn.BatchNorm2d(out_f // 8),
            nn.ReLU(),
            nn.Conv2d(out_f // 8, out_f, 1),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
        )
        # 降低特征维度
        self.last_conv = nn.Sequential(
            nn.Conv2d(out_f, out_f // 8, 1),
            nn.BatchNorm2d(out_f // 8),
            nn.ReLU(),
            nn.Conv2d(out_f // 8, 1, 1),
        )

    def forward(self, x, adj, e):
        """

        :param x: (b, t, n, f)
        :param adj: (n, n)
        :param e: (b, num_pred, 2)
        :return: (b, num_pred, n, 1)
        """
        x = self.first_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x1 = self.stblock1(x, adj, e)
        x2 = self.stblock2(x1, adj, e)
        y = x1 + x2
        return self.last_conv(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)


if __name__ == "__main__":
    a = torch.rand(64, 12, 32, 3)
    print(a.shape)
    e = torch.rand(64, 6, 2)
    adj = torch.rand(32, 32)
    net = AG_JNet(3, 64, 4, 4, 32, num_his=12, num_pred=6)
    y = net(a, adj, e)
    print(y.shape)
