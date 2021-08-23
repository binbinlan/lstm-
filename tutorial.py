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
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 1000, 800
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('土壤含水率LSTM预测模型')  # 窗口命名

    canvas = tk.Label(win,text='中国科学院水利部成都山地灾害与环境研究所')
    canvas.pack()

    canvas2=tk.Label(win,text='Binbinlan')
    canvas2.pack(side=BOTTOM,anchor=CENTER)

    var = tk.StringVar()  # 创建变量文字
    var.set('↓请选择训练数据集↓')

    photo1=PhotoImage(file=r'./pic/brand2.png')

    imgLabel1=Label(win,image=photo1)
    imgLabel1.pack(side=tk.TOP)#,anchor=NW)

    tk.Label(win, textvariable=var, bg='#C1FFC1', font=('宋体', 21), width=20, height=2).pack()
    tk.Button(win, text='选择数据集', width=20, height=2, bg='#FF8C00', command=lambda: getdata(var, canvas2),
              font=('圆体', 10)).pack()

    L1 = tk.Label(win, text="选择你需要的 列(请用空格隔开，从0开始）")
    L1.pack()
    E1 = tk.Entry(win, bd=5)
    E1.pack()

    L2 = tk.Label(win, text="请设置时序步长")
    L2.pack()
    E2 = tk.Entry(win, bd=5)
    E2.pack()

    L3 = tk.Label(win, text="请设置训练集大小")
    L3.pack()
    E3 = tk.Entry(win,bd=5)
    E3.pack()


    button1 = tk.Button(win, text="提交",command=lambda:[getLable(E1),print(canvas2.text),changeLabel(var)])
    button1.pack()

    tk.Button(win, text='完成', width=20, height=2, bg='#FF8C00', command=lambda:[gethours(E2,E3),
                                                                                main(data,N_hours,N_train_hours),
                                                                                win.quit()],
              font=('圆体', 10)).place(anchor=CENTER,x=500,y=500)


    win.mainloop()

def getdata(var, canvas):
    global file_path
    file_path = filedialog.askopenfilename()
    var.set("注，最后一个为label")
    # 读取文件第一行标签
    with open(file_path, 'r', encoding='gb2312') as f:
    #with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取所有行
        data2 = lines[0]
    #print()

    canvas.configure(text=data2)
    canvas.text = data2

def getLable(E1):
    string = E1.get()
    print('训练数据索引号为（从0开始）：',string)
    gettraindata(string)

def changeLabel(var):
    var.set('请点击完成')

def gettraindata(string):
    f_open = open(file_path)
    df = pd.read_csv(f_open)  # 读入股票数据
    list = string.split()
    print(list)
    x = len(list)
    index=[]
    # data = df.iloc[:, [1,2,3]].values  # 取第3-10列 （2:10从2开始到9）
    for i in range(x):
        q = int(list[i])
        index.append(q)
    global data
    data = df.iloc[:, index].values
    print(data)
    #main(data)

def gethours(E2,E3):
    global N_hours
    global N_train_hours
    N_train_hours=int(E3.get())
    N_hours=int(E2.get())


def main(data,N_hours,N_train_hours):       #N_train_hours为训练数据量
    N_features = len(data[0])
    print('训练序列维度：',N_features,'\n')
    print('训练时间步长：',N_hours,'\n')
    print('训练集大小为：',N_train_hours,'\n')
    training(data,N_hours,N_features,N_train_hours)
    pass

if __name__ == '__main__':
    creat_windows()


