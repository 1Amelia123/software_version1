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
    print(s_ls1)
    temp=s_ls1[0][1]
    count=1
    # break
    for i in range(1,639):
        # 接下来我们来按月分
        if(s_ls1[i][0].year==2016):
            continue
        else:
            if(s_ls1[i][0].month==s_ls1[i-1][0].month):
                temp=temp+s_ls1[i][1]
                count=count+1
            else:
                d_c.append(temp/count)
                # print(temp,count)
                temp=s_ls1[i][1]
                count=1
        '''
        # 如果是直接按天分那就直接是得到得这个结果
        d_c.append(s_ls1[i][1])
        '''

    Shu.append(d_c)
Shu=np.array(Shu,dtype=np.float32)
# np.save('year_pre_none.npy',Shu[0:10])
print(Shu)
# 聚类实现

from sklearn.cluster import KMeans

# 定义候选的K值。
scope = range(1, 15)
# 定义SSE列表，用来存放不同K值下的SSE。
sse = []
s = StandardScaler()
Shu = s.fit_transform(Shu)
# np.save('year_pre.npy',Shu[0:10])
# print(np.isnan(mos).any())
for k in scope:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Shu)
    sse.append(kmeans.inertia_)
plt.xticks(scope)
sns.lineplot(scope, sse, marker="o")
plt.show()
estimator = KMeans(n_clusters=4)
estimator.fit(Shu)
estimator.predict(Shu)

label_pred = estimator.labels_  # 获取聚类标签
joblib.dump(estimator.fit(Shu),"电力用户集群模型.mkl")
estimator=joblib.load("电力用户集群模型.mkl")
df=pd.DataFrame(estimator.cluster_centers_)
df.to_excel('center.xls')
df2=pd.read_excel('center.xls')
label_pred=estimator.predict(Shu)
l1=0
l2=1
l3=0
l4=0
for i in range(len(label_pred)):#遍历每一个用户
    if label_pred[i] == 0:
        l1=l1+1
        plt.plot(range(len(Shu[i])), Shu[i], color='purple',shape='o')
    elif label_pred[i] == 1:
        l2=l2+1
        plt.plot(range(len(Shu[i])), Shu[i], color='b',shape='o')
    elif label_pred[i] == 2:
        l3=l3+1
        plt.plot(range(len(Shu[i])), Shu[i], color='violet',shape='o')
    else:
        l4=l4+1
        plt.plot(range(len(Shu[i])), Shu[i], color='seagreen',shape='o')

plt.title('用户负荷曲线聚类示意图')
plt.show()
print(l1,l2,l3,l4)