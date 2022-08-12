import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 模型系数
input_dim = 7  # 数据的特征数
hidden_dim = 64  # 隐藏层的神经元个数
num_layers = 1  # LSTM的层数
output_dim = 1  # 预测值的特征数
pre_days = 7  # 以1周的数据为一组
seq = 2 # 用1周前的数据去预测现在数据
batch_size = 128 # 每组数据量
num_epochs = 100 # 模型遍历次数

#读数据
df_main = pd.read_csv('data_1.csv')
sel_col = ["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
for col in sel_col:
    df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1, 1))
df_main['target'] = df_main['power_consumption'].shift(-1)


# LSTM模型 输入为7输出为1 32个神经元
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.5)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out


# 创建两个列表，用来存储数据的特征和标签
data_feat, data_target = [], []

for index in range(len(df_main) - seq):
    # 构建特征集
    data_feat.append(df_main[["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]][
                     index: index + seq].values)
    # 构建target集
    data_target.append(df_main['target'][index:index + seq])

# 将特征集和标签集整理成numpy数组
data_feat = np.array(data_feat)
data_target = np.array(data_target)
# 这里按照8:2的比例划分训练集和测试集
test_set_size = 122  # np.round(1)是四舍五入，
train_size = data_feat.shape[0] - (test_set_size)


trainX = torch.from_numpy(data_feat[:train_size].reshape(-1, seq, 7)).type(torch.Tensor)
testX = torch.from_numpy(data_feat[train_size:].reshape(-1, seq, 7)).type(torch.Tensor)
trainY = torch.from_numpy(data_target[:train_size].reshape(-1, seq, 1)).type(torch.Tensor)
testY = torch.from_numpy(data_target[train_size:].reshape(-1, seq, 1)).type(torch.Tensor)
print('x_train.shape = ', trainX.shape)
print('y_train.shape = ', trainY.shape)
print('x_test.shape = ', testX.shape)
print('y_test.shape = ', testY.shape)

# dataloader读入数据
train = torch.utils.data.TensorDataset(trainX, trainY)
test = torch.utils.data.TensorDataset(testX, testY)
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

# 封装模型
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化算法
# 模型评估条件
loss_fn = torch.nn.MSELoss(size_average=True)  # 使用均方差作为损失函数


# # 打印模型各层的参数尺寸
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

# 训练模型
hist = np.zeros(num_epochs)
ls = []
for t in range(num_epochs):
    y_train_pred = model(trainX)
    loss = loss_fn(y_train_pred, trainY)
    ls.append(loss.item())
    if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    # 梯度归零
    optimiser.zero_grad()
    # Backward
    loss.backward()
    # 更新参数
    optimiser.step()

# 模型预测
y_test_pred = model(testX)
list1 = []
list2 = []
y_test_pred = model(testX)
loss1 = loss_fn(y_test_pred[:-pre_days], testY[:-pre_days]).item()
print(loss1)

# 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
# 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
# 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
pred_value = y_train_pred.detach().numpy()[:, -1, 0]
true_value = trainY.detach().numpy()[:, -1, 0]


# 绘图
plt.plot(pred_value, label="Preds")  # 预测值
plt.plot(true_value, label="Data")  # 真实值
plt.legend()
plt.show()
pred_value = y_test_pred.detach().numpy()[:, -1, 0]
true_value = testY.detach().numpy()[:, -1, 0]

pred_value = scaler.inverse_transform(pred_value.reshape(-1, 1))
true_value = scaler.inverse_transform(true_value.reshape(-1, 1))
pred_value = pred_value
print(pred_value)
print(true_value)

plt.plot(pred_value, label="Preds")  # 预测值
plt.plot(true_value, label="Data")  # 真实值
plt.legend()
plt.show()

plt.plot(ls, label='loss')
plt.legend()
plt.xlabel = '迭代次数'
plt.ylabel = '损失值'
plt.title("loss vs epoch")
plt.show()

# 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，即是二进制方式,DT是模型名字
pickle.dump(model,open("企业电力营销模型.mkl","wb"))   # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
