import pandas as pd
import numpy as np
path = r'data.xlsx'
df = pd.read_excel(path,usecols=None)   # 直接使用 read_excel() 方法读取, 不读取列名
lines = df.values.tolist()
print(len(lines))
result = []
username = '1000000001'
count = 0
money = 0
i = 0
ls = []
for line in lines:
    i += 1
    if i != len(lines)-1:
        if str(line[0]) == username:
            count = count + 1
            money += int(line[2])
        else:
            ls.append(username)
            ls.append(count)
            ls.append(money)
            result.append(ls)
            username = str(line[0])
            ls = []
            count = 1
            money = int(line[2])
    else:
        ls.append(username)
        ls.append(count)
        ls.append(money)
        result.append(ls)


print(len(result))
print(result)


# 数据写入
# -*- coding: utf-8 -*-
import pandas as pd


def pd_toExcel(data, fileName):  # pandas库储存数据到excel
    ids = []
    counts = []
    prices = []
    for i in range(len(data)):
        ids.append(data[i][0])
        counts.append(data[i][1])
        prices.append(data[i][2])

    dfData = {  # 用字典设置DataFrame所需数据
        '用户编号': ids,
        '缴费次数': counts,
        '缴费金额': prices
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）


fileName = 'users.xlsx'
pd_toExcel(result, fileName)
