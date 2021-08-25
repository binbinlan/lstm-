import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import time
from train import training

N_hours = 0
N_train_hours = 0


def creat_windows():

    win = tk.Tk()  # 创建窗口
    # sw = win.winfo_screenwidth()
    # sh = win.winfo_screenheight()
    # ww, wh = 1000, 800
    # x, y = (sw - ww) / 2, (sh - wh) / 2
    # win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('土壤含水率LSTM预测模型')  # 窗口命名

    canvas = tk.Label(win,text='中国科学院水利部成都山地灾害与环境研究所',font=('雅黑', 15))
    canvas.grid(column=17)
    Label(win,text='4').grid(row=8)
    Button(win, text='123').grid(row=5, column=6)

    win.mainloop()

creat_windows()