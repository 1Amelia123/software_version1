import pandas as pd
import numpy as np

# 任务1
path = r'users.xlsx'
df = pd.read_excel(path,usecols=None)   # 直接使用 read_excel() 方法读取, 不读取列名
lines = df.values.tolist()
print(len(lines))
result = []
counts = 0
money = 0
i = 0
ls = []
for line in lines:
    counts += int(line[1])
    money += int(line[2])
avg_counts = counts/100
avg_money = money/100

print("平均缴费次数：",avg_counts)
print("平均缴费金额：",avg_money)

# 数据写入
import csv
#  创建csv文件对象，u是写入中文，encoding='utf-8'是设置编码格式，newline=''为了防止空行
f = open(u'居民客户的用电缴费习惯分析 1.csv', 'w', encoding='utf-8-sig', newline='')
csv_write = csv.writer(f)
csv_write.writerow(['平均缴费金额', '平均缴费次数'])
csv_write.writerow([avg_money, avg_counts])
# --------------------------------------------------
# 任务2
type = ""
for line in lines:
    ls = []
    if line[1] <= avg_counts:
        if line[2] <= avg_money:
            type = "低价值型客户"
            ls.append(str(line[0]))
            ls.append(type)
        else:
            type = "潜力型客户"
            ls.append(str(line[0]))
            ls.append(type)
    else:
        if line[2] <= avg_money:
            type = "大众型客户"
            ls.append(str(line[0]))
            ls.append(type)
        else:
            type = "高价值型客户"
            ls.append(str(line[0]))
            ls.append(type)
    result.append(ls)
print(result)

# 数据写入
import csv
#  1.创建csv文件对象，encoding='utf-8'是设置编码格式，newline=''为了防止空行
f = open(u'居民客户的用电缴费习惯分析 2.csv', 'w', encoding='utf-8-sig', newline='') #居民客户的用电缴费习惯分析 2
#  2.基于文件对象构建csv写入对象
csv_write = csv.writer(f)
#  3.构建列表头
csv_write.writerow(['用户编号', '客户类型'])
for data in result:
    #  4.写入csv文件
    csv_write.writerow([str(data[0]), data[1]])
