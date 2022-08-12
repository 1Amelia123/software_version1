# 读取数据
import pandas as pd
import numpy as np
from tqdm import tqdm

train_data = pd.read_csv("C:/Users/86147/PycharmProjects/软件杯/二/dataset/train.csv")
holiday = pd.read_csv("C:/Users/86147/PycharmProjects/软件杯/二/dataset/holiday.csv")
weather = pd.read_csv("C:/Users/86147/PycharmProjects/软件杯/二/dataset/yangzhong.csv", header=None)

train_data['record_date'] = pd.to_datetime(train_data['record_date'])
holiday['date'] = pd.to_datetime(holiday['date'])
weather[0] = pd.to_datetime(weather[0])
# train_data.insert(train_data.shape[1], 'holiday', 0)
# train_data.insert(train_data.shape[1], 'weather', 0)

# train_data.rename(columns={'record_date': 'date'})
# weather.rename(columns={0: 'date'})
train_data.columns = ['date', 'user_id', 'power_consumption']
weather.columns = ['date', 'low_temp', 'high_temp', 'kind', 'wind', 'level']
# print(train_data)
# print(holiday)
# print(weather)
train_data = pd.merge(train_data,weather,on='date',how='left')
train_data = pd.merge(train_data,holiday,on='date',how='left')
print(train_data)
train_data.to_csv("train_data.csv",index=False,encoding='utf-8-sig')
