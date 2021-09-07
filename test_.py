import tkinter as tk
import tkinter.simpledialog
from tkinter import *
from tkinter import filedialog

import pandas
import pandas as pd
import time
from train import training
from pandas import read_csv
from matplotlib import pyplot
from matplotlib import font_manager as fm, rcParams
import matplotlib as plt

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
    filepath=filedialog.askopenfilename()
    data = pd.read_csv(filepath,encoding='gb2312')

    data=data.values
    cols=data.shape[1]
    y_real=data[:,cols-1]
    print(type(data))
    print(data)
    print(y_real)
    var.set('准备就绪')

def test():
    pass



