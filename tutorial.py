import tkinter as tk
from tkinter import *
from tkinter import filedialog
import time

def getdata(var, canvas):
    global file_path
    file_path = filedialog.askopenfilename()
    var.set("注，最后一个为label")
    # 读取文件第一行标签
    with open(file_path, 'r', encoding='gb2312') as f:
    # with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取所有行
        data2 = lines[0]
    print()

    canvas.configure(text=data2)
    canvas.text = data2


win = tk.Tk()  # 创建窗口
sw = win.winfo_screenwidth()
sh = win.winfo_screenheight()
ww, wh = 1200, 1000
x, y = (sw - ww) / 2, (sh - wh) / 2
win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

win.title('土壤含水率LSTM预测模型')  # 窗口命名

canvas = tk.Label(win,text='中国科学院水利部成都山地灾害与环境研究所')
canvas.pack()

var = tk.StringVar()  # 创建变量文字
var.set('请选择训练数据集')
photo1=PhotoImage(file=r'./pic/brand2.png')
imgLabel1=Label(win,image=photo1)
imgLabel1.pack(side=tk.TOP)#,anchor=NW)
tk.Label(win, textvariable=var, bg='#C1FFC1', font=('宋体', 21), width=20, height=2).pack()
tk.Button(win, text='选择数据集', width=20, height=2, bg='#FF8C00', command=lambda: getdata(var, canvas),
          font=('圆体', 10)).pack()

win.mainloop()


