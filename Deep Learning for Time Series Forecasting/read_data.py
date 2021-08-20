from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
i2 = 0
# plot each column
pyplot.figure()
#columns2=['rain','temperature','speed','NDVI','DEM','type','water']
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	#pyplot.title(columns2[i2], y=0.5, loc='right')
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
	i2 += 1
pyplot.show()

'''
1.No 行数
2.year 年
3.month 月
4.day 日
5.hour 小时
6.pm2.5 PM2.5浓度
7.DEWP 露点
8.TEMP 温度
9.PRES 大气压
10.cbwd 风向
11.lws 风速
12.ls 累积雪量
13.lr 累积雨量
'''

