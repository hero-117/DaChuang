import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 用于画图
from sklearn.preprocessing import MinMaxScaler  # 用于数据标准化
import os
import pandas as pd
import numpy as np


def loadBoston():  # 从文件中读取数据
    NHT_data_path = os.path.join(os.getcwd(), 'mean.xlsx')  # 在绿色处输入文件名称
    NHT_xlsx = pd.read_excel(NHT_data_path, index_col=0)
    NHT_CSV = os.path.join(os.getcwd(), 'data.csv')
    NHT_xlsx.to_csv(NHT_CSV, encoding='utf-8')
    data_csv = pd.read_csv(NHT_CSV, header=None, dtype=np.float32)
    # print(data_csv)
    return data_csv


def split_train_test_from_df(df, test_ratio=0.2):  # 从数据集中分割 训练集 和 测试集
    test_df = df.sample(frac=test_ratio)
    train_df = df[~df.index.isin(test_df.index)]
    return train_df, test_df


def preprocess(df):
    ss = MinMaxScaler()
    df = ss.fit_transform(df)
    df = pd.DataFrame(df)
    return df, ss.data_min_[-1], ss.data_max_[-1]


def anti_minMaxScaler(d, min, max):  # 数据归一化
    '''
        (x-min)/(max-min)
    '''
    return d * (max - min) + min


def drawLines(ds, names):  # 画出预测线和标准线
    '''
    :param ds: [数据列表1，数据列表2]
    :param names:[name1,name2]
    :return:
    '''
    plt.figure()
    x = range(len(ds[0]))
    for d, name in zip(ds, names):
        plt.plot(x, d, label=name)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.legend(fontsize=16, loc='upper left')
        plt.grid(c='gray')
    plt.savefig(fname='test.png', dpi=200)
    # plt.show()


class Linear_Reg(nn.Module):  # 定义神经网络模型参数

    def __init__(self, n_features):
        super(Linear_Reg, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)  # 输入维度，输出维度，是否有偏置

    def forward(self, x):
        y = self.linear(x)
        y = torch.squeeze(y)  # 转变为一维向量
        return y


def calculate_accuracy(true, predict):
    len_true = np.shape(true)[0]
    loss0 = np.zeros(len_true)
    # print(loss0)
    for i in range(0, len_true):
        loss0[i] = 1 - (abs(predict[i] - true[i]) / true[i])
    loss_mean = np.mean(loss0)
    print('平均正确率为：{:.2%}'.format(loss_mean))


@torch.no_grad()
def eva(dates, net, min, max):  # 定义测试（生成测试集，输出拟合图像）
    net.eval()
    X = torch.Tensor(dates[:, :-1])
    y = dates[:, -1]
    y_pred = net(X)
    y_pred = anti_minMaxScaler(y_pred, min, max)
    y = anti_minMaxScaler(y, min, max)
    criterion = torch.nn.MSELoss()  # 平方差损失函数
    loss = criterion(y_pred, torch.Tensor(y))
    # print(loss ** 0.5)
    y_pred_np0 = y_pred
    y_pred_np1 = y_pred_np0.numpy()
    # print(np.shape(y_pred_np1))
    # print(np.shape(y))
    drawLines([y, y_pred], ['true', 'predict'])
    calculate_accuracy(y, y_pred_np1)

    # plt.drawLines(abs(y - y_pred) / y)


def train(epochs=100, batchSize=16, lr=0.01):  # 定义训练

    df, min, max = preprocess(loadBoston())
    train_df, test_df = split_train_test_from_df(df, test_ratio=0.2)

    net = Linear_Reg(train_df.shape[1] - 1)  # 初始化线性回归模型
    criterion = torch.nn.MSELoss()  # 平方差损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 随机梯度下降 采用SGD优化器
    net.train()
    for e in range(epochs):
        for datas in DataLoader(train_df.values, batch_size=batchSize, shuffle=True):
            optimizer.zero_grad()  # 梯度归0
            X = datas[:, :-1]  # 获取X
            y = datas[:, -1]  # 获取y
            y_pred = net(X)  # 得到预测值y
            loss = criterion(y_pred, y)  # 将预测的y与真实的y带入损失函数计算损失值
            loss.backward()  # 后向传播
            optimizer.step()  # 更新所有参数
        print('epoch {},loss={:.4f}'.format(e, loss))

    torch.save(net, 'Ml_model')

    eva(test_df.values, net, min, max)


if __name__ == '__main__':
    train()
