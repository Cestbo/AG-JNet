#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 15:19
# @Author  : Cestbo
# @Site    : 
# @File    : main.py
# @Software: PyCharm

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from model_nj import AG_JNet
from stgcn import STGCN
from torch.utils.data import DataLoader, TensorDataset
import utils
import torch
import os
from time import time
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False, help='Test directly if there are trained model')
arg = parser.parse_args()

# 读取参数
config = configparser.ConfigParser()
config.read('config.ini')

net_setting = config['net']
net_name = net_setting['net_name']
in_feature = int(net_setting['in_feature'])
dim_exp = int(net_setting['dim_exp'])
num_his = int(net_setting['num_his'])
num_pred = int(net_setting['num_pred'])
layers_sm = int(net_setting['layers_sm'])
layers_tm = int(net_setting['layers_tm'])

dataset_setting = config['dataset']
dataset = dataset_setting['dataset']
train_split = float(dataset_setting['train_split'])
val_split = float(dataset_setting['val_split'])

train_setting = config['train']
batch_size = int(train_setting['batch_size'])
lr = float(train_setting['lr'])
epoches = int(train_setting['epoch'])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

print('参数设置信息：')
print(f' num_pred:{num_pred}\t dataset:{dataset}\n net_name:{net_name}\t device:{device}')

x, y, e = utils.get_data(f"data/{dataset}/{dataset}.npz", num_his, num_pred)
x, y, e = x.to(device), y.to(device), e.to(device)
n = x.shape[2]
adj = utils.get_adj("data/{}/distance.csv".format(dataset), n).to(device)
# adj = None

if not os.path.exists(f'experiment/{net_name}'):
    os.mkdir(f'experiment/{net_name}')
net = AG_JNet(in_feature, dim_exp, layers_sm, layers_tm, n, num_his, num_pred, device).to(device)
if net_name == "STGCN":
    net = STGCN(n, in_feature, num_his, num_pred).to(device)


num_params = sum(param.numel() for param in net.parameters())
print('模型参数量：', num_params)
utils.init_net(net)

criterion = nn.MSELoss().to(device)
opt = Adam(net.parameters(), lr=lr)

num_sample = x.shape[0]
split_index = int(train_split * num_sample)
split_index1 = int(val_split * num_sample)
train_x, train_y, val_x, val_y, test_x, test_y, train_e, val_e, test_e, mean, std \
    = utils.normalization(x, y, e, split_index, split_index1)

# 必须打乱数据集，不然loss降不下来。
train_loader = DataLoader(dataset=TensorDataset(train_x, train_e, train_y), batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(dataset=TensorDataset(val_x, val_e, val_y), batch_size=batch_size,
                        shuffle=False)
test_loader = DataLoader(dataset=TensorDataset(test_x, test_e, test_y), batch_size=batch_size,
                         shuffle=False)

train_losses = []
val_losses = []


def test():
    print("------------test------------")
    test_mae = 0
    test_rmse = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # t = time()
            x, e, y = data
            output = net(x, adj, e)
            # print(time() - t)
            test_mae += utils.mae(output, y)
            test_rmse += utils.rmse(output, y)

        print("mae:{:2f} , rmse:{:2f}".format(test_mae/(i+1), test_rmse/(i+1)))



def train():
    for epoch in range(epoches):
        net.train()
        train_count = 0
        train_mae = 0
        sum_train_loss = 0
        with tqdm(total=len(train_loader), desc='TRAINING-epoch-{}'.format(epoch), unit='batches') as bar:
            for i, data in enumerate(train_loader):
                forward_t = time()
                x, e, y = data
                output = net(x, adj, e)
                # output = net(x, adj)
                backward_t = time()
                loss = criterion(output, y)
                sum_train_loss += loss.item()
                # print("前向传播时间：", backward_t - forward_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_mae += utils.mae(output, y)
                bar.set_postfix(loss=f'{sum_train_loss / (i + 1):.2f}', mae=f'{train_mae / (i + 1):.2f}')
                bar.update()
                train_count += 1
                # print("反向传播时间：", time() - backward_t)
                # print("batch时间", time()-forward_t)
        train_losses.append(sum_train_loss / train_count)

        with torch.no_grad():
            net.eval()
            val_count = 0
            val_mae = 0
            sum_val_loss = 0
            with tqdm(total=len(val_loader), desc='VALIDATION-epoch-{}'.format(epoch), unit='batches') as bar:
                val_t = time()
                for i, data in enumerate(val_loader):
                    x, e, y = data
                    output = net(x, adj, e)
                    # output = net(x, adj)
                    loss = criterion(output, y)
                    val_count += 1
                    val_mae += utils.mae(output, y)
                    sum_val_loss += loss.item()
                    bar.set_postfix(loss=f'{sum_val_loss / (i + 1):.2f}', val_mae=f'{val_mae / (i + 1):.2f}')
                    bar.update()
                # print("测试时间：", time()-val_t)
            val_losses.append(sum_val_loss / val_count)

        if (sum_val_loss / val_count) <= min(val_losses):
            print("epoch-{:d}  保存模型。。。。。。".format(epoch))
            torch.save(net, f"experiment/{net_name}/net_{dataset}_{num_pred}.pkl")

    test()
    dict = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    return dict


if __name__ == '__main__':
    if not arg.test:
        loss_dict = train()
        # utils.visualization(**loss_dict)
    net = torch.load(f"experiment/{net_name}/net_{dataset}_{num_pred}.pkl", map_location="cuda:0")
    test()
