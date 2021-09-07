import tkinter as tk
import tkinter.simpledialog
from tkinter import *
from tkinter import filedialog
from tensorflow.python.keras.models import load_model
import pandas
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
#from keras.models import Sequential
from tensorflow.python.keras.models import Sequential
#from keras.layers import Dense
from tensorflow.python.keras.layers.core import Dense
#from keras.layers import LSTM
from tensorflow.python.keras.layers.recurrent import LSTM
#from tensorflow.python.keras.layers.core import LSTM
import time
from train import training
from pandas import read_csv
from matplotlib import pyplot
from matplotlib import font_manager as fm, rcParams
import matplotlib as plt
import tutorial
import globalvar as gl
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def creat_test():
    win = tk.Tk()  # 创建窗口
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 1000, 1000
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('土壤含水率LSTM预测模型')  # 窗口命名

    canvas = tk.Label(win,text='中国科学院水利部成都山地灾害与环境研究所',font=('雅黑', 18),pady=20)
    canvas.pack()


    var = tk.StringVar()  # 创建变量文字
    var.set('↓请选择模型文件↓')

    photo1=PhotoImage(file=r'./pic/brand3.png')

    imgLabel1=Label(win,image=photo1)
    imgLabel1.pack(side=tk.TOP,pady=20)#,anchor=NW)

    tk.Label(win, textvariable=var, bg='#abe4ab', font=('仿宋', 21), width=20, height=2).pack(pady=15)
    tk.Button(win, text='选择模型', width=20, height=2, bg='#FF8C00', relief=RAISED, borderwidth=3,
              command=lambda: getmodel(var),
              font=('圆体', 10)).pack(pady=10)


    tk.Button(win, text='选择测试集', width=20, height=2, bg='#FF8C00', relief=RAISED, borderwidth=3,
              command= lambda:get_test(var),
              font=('圆体', 10)).pack(pady=10)

    tk.Button(win, text='验证', width=20, height=2, bg='#FF8C00', relief=RAISED, borderwidth=3,
              command= test,
              font=('圆体', 10)).pack(pady=10)
    win.mainloop()
def getmodel(var):
    global model_path
    model_path = filedialog.askopenfilename()
    var.set("选择测试集")

def get_test(var):

    global data
    filepath=filedialog.askopenfilename()
    data = pd.read_csv(filepath,encoding='gb2312')
    string=gl.get_value('送训列')
    list = string.split()
    print(list)
    x = len(list)
    index=[]
    # data = df.iloc[:, [1,2,3]].values  # 取第3-10列 （2:10从2开始到9）
    for i in range(x):
        q = int(list[i])
        index.append(q)

    data = data.iloc[:, index].values
    print(data)
    #main(data)

    var.set('准备就绪')

def test():
    model=load_model(model_path)
    cols = data.shape[1]
    y_real = data[:, cols - 1]
    print(type(data))
    print(data)
    print(y_real)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)


    n_hours = gl.get_value('步长')
    n_features = data.shape[1]
    print(n_hours, n_features)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    #reframed = series_to_supervised(values, n_hours, 1)
    print(reframed.shape)
    print(reframed.head(5))

    # split into train and test sets
    values = reframed.values
    #n_train_hours = 365 * 24
    test = values[:, :]

    # split into input and outputs
    n_obs = n_hours * n_features

    test_X, test_y = test[:, :n_obs], test[:, -1]
    print(test_X.shape, len(test_y), test_X.shape)

    # reshape input to be 3D [samples, timesteps, features]

    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(test_X.shape, test_y.shape)
#creat_test()



