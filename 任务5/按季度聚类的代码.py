import pandas as pd
import  numpy as np
import matplotlib
matplotlib.use("TKAgg")
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('TkAgg')
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#
data = pd.read_csv('train.csv')  # 读取负荷
data.loc['record_date']=pd.to_datetime(data['record_date'])#将record_date转换为时间格式
data=data.dropna()#删除空值

data1=data.groupby("user_id")#根据用户ID进行分组，返回的data1是一个对象,object

Shu = []
for name, group in data1:#name 是user_id ,group该用user_id==name 时的其他项
    # print(name)
    # print(group.shape[0])
    if(group.shape[0]!=639):
        continue
        # print(name)
    ls1=[]
    dv=group['record_date'].values
    # print(dv[0])
    pv=group['power_consumption'].values
    # print(group['record_date'].values)
    for i in range(639):
        ls1_1=[]
        temp=datetime.datetime.strptime(dv[i],'%Y/%m/%d').date()
        ls1_1.append(temp)
        ls1_1.append(pv[i])
        ls1.append(ls1_1)
    s_ls1=sorted(ls1)
    d_c=[]
    # break
    for i in range(1,639):
        # 更改这里的值,能够得到对应季度的模型,当值为1,2,3时为第一季度,当值为4,5,6时为第二季度,当值为7,8,9时为第三季度,当值为10,11,12时为第三季度
        if((s_ls1[i][0].year==2015 and s_ls1[i][0].month==12) or(s_ls1[i][0].year==2015 and s_ls1[i][0].month==11) or (s_ls1[i][0].year==2015 and s_ls1[i][0].month==10)):
            d_c.append(s_ls1[i][1])
    Shu.append(d_c)
Shu=np.array(Shu,dtype=np.float32)
# np.save('jidu4_pre_none.npy',Shu[0:10])
print(Shu)
# 聚类实现
# 定义候选的K值。
scope = range(1, 15)
# 定义SSE列表，用来存放不同K值下的SSE。
sse = []
# 对数据进行归一化处理
s = StandardScaler()
Shu = s.fit_transform(Shu)
# np.save('jidu4_pre.npy',Shu[0:10])
for k in scope:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Shu)
    sse.append(kmeans.inertia_)
plt.xticks(scope)
sns.lineplot(scope, sse, marker="o")
plt.title("第四季度")
plt.show()
# 底下是预测
estimator = KMeans(n_clusters=6)
estimator.fit(Shu)
estimator.predict(Shu)
# label_pred = estimator.labels_  # 获取聚类标签
joblib.dump(estimator.fit(Shu),"电力用户集群模型_1.mkl")
estimator=joblib.load("电力用户集群模型_1.mkl")
label_pred=estimator.predict(Shu)
l1=0
l2=0
l3=0
l4=0
l5=0
l6=0
for i in range(len(label_pred)):#遍历每一个用户
    if label_pred[i] == 0:
        l1=l1+1
        plt.plot(range(len(Shu[i])), Shu[i], color='r',label='第一类',alpha=0.5)
    elif label_pred[i] == 1:
        l2=l2+1
        plt.plot(range(len(Shu[i])), Shu[i], color='b',label='第二类',alpha=0.5)
    elif label_pred[i] == 2:
        l3=l3+1
        plt.plot(range(len(Shu[i])), Shu[i], color='violet',label='第三类',alpha=0.5)
    elif label_pred[i]==4:
        l5=l5+1
        plt.plot(range(len(Shu[i])), Shu[i], color='yellow',label='第四类',alpha=0.5)
    elif label_pred[i] == 5:
        l6 = l6 + 1
        plt.plot(range(len(Shu[i])), Shu[i], color='purple',label='第五类',alpha=0.5)
    else:
        l4=l4+1
        plt.plot(range(len(Shu[i])), Shu[i], color='seagreen',label='第六类',alpha=0.5)

plt.title('第四季度用户负荷曲线聚类示意图')
plt.show()
# print(l1,l2,l3,l4,l5,l6)


