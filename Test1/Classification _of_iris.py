import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore import Model

import os
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import genfromtxt


class my_net(nn.Cell):
    def __init__(self):
        super(my_net, self).__init__()
        self.fc1 = nn.Dense(4, 10)
        self.fc2 = nn.Dense(10, 3)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 加载数据集
iris_data = genfromtxt('iris.csv', delimiter=',')
print(iris_data[:10])

# 数据划分
iris_data = iris_data[1:]
X = iris_data[:, :4].astype(np.float32)
y = iris_data[:, -1].astype(np.int32)
X /= np.max(np.abs(X), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#
train_data = (X_train, y_train)
train_data = ds.NumpySlicesDataset(train_data)

test_data = (X_test, y_test)
test_data = ds.NumpySlicesDataset(test_data)

train_data = train_data.batch(32)
test_data = test_data.batch(32)

print(type(train_data))

net = my_net()
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True)
lr = 0.01  # 学习率
momentum = 0.9  # 动量
net_opt = nn.Momentum(net.trainable_params(), lr, momentum)

model = Model(net, net_loss, net_opt, metrics={"accuracy": Accuracy()})

model_output=model.train(10, train_data)
model_evaluate=model.eval(test_data)
print(model_evaluate)
