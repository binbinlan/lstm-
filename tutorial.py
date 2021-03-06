import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import time
from train import training
from pandas import read_csv
from matplotlib import pyplot
from matplotlib import font_manager as fm, rcParams
import matplotlib as plt

#定义初始化参数


N_hours = 0
N_train_hours = 0
file_path=''


def creat_windows():

    win = tk.Tk()  # 创建窗口
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 1000, 830
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('土壤含水率LSTM预测模型')  # 窗口命名

    canvas = tk.Label(win,text='中国科学院水利部成都山地灾害与环境研究所',font=('雅黑', 18),pady=20)
    canvas.pack()


    var = tk.StringVar()  # 创建变量文字
    var.set('↓请选择训练数据集↓')

    photo1=PhotoImage(file=r'./pic/brand3.png')

    imgLabel1=Label(win,image=photo1)
    imgLabel1.pack(side=tk.TOP,pady=20)#,anchor=NW)

    tk.Label(win, textvariable=var, bg='#abe4ab', font=('仿宋', 21), width=20, height=2).pack(pady=15)
    tk.Button(win, text='选择数据集', width=20, height=2, bg='#FF8C00', relief=RAISED, borderwidth=3,
              command=lambda: getdata(var, canvas2),
              font=('圆体', 10)).pack()

    L1 = tk.Label(win, text="选择你需要的 列(请用空格隔开，从0开始)：")
    L1.pack(pady=10)
    E1 = tk.Entry(win,width=40, bd=5)
    E1.pack()

    L2 = tk.Label(win, text="请设置时序步长：")
    L2.pack(pady=10)
    E2 = tk.Entry(win,bd=5)
    E2.pack()

    L3 = tk.Label(win, text="请设置训练集大小：")
    L3.pack(pady=10)
    E3 = tk.Entry(win,bd=5)
    E3.pack()


    button1 = tk.Button(win, text="提交到网络",width=20, height=2,
                        bg='#FF8C00',relief=RAISED, borderwidth=3,
                        font=('圆体', 10),
                        command=lambda:[getLable(E1),
                                        print(canvas2.text),
                                        changeLabel_1(var,checkVar0)])
    button1.pack(pady=15)



    tk.Button(win, text='开始训练', width=20, height=2, bg='#FF8C00',
              command=lambda:[gethours(E2,E3),
                              win.iconify(),
                 #timeset(),
                main(var,data,N_hours,N_train_hours)],
              relief=RAISED, borderwidth=3,
              font=('圆体', 10)).pack(pady=0)


    group=LabelFrame(win,text='其他选项',pady=10)
    group.pack(pady=10)

    # Choice=[
    #     ('显示数据集',1),
    #     ('保存模型',2)
    # ]
    #
    # v=IntVar()
    # for choice,num in Choice:
    #     b=Radiobutton(group,text=choice,variable=v,value=num)
    #     b.pack()

    checkVar0=StringVar(value=1)
    check = Checkbutton(group,text='显示训练集',variable=checkVar0)
    check.pack()

    # def button_click(event=None):
    #     print(checkVar0.get())

    # b1 = Button(group,text='click me ',command=button_click)
    # b1.pack()


    canvas2=tk.Label(win,text='Binbinlan')
    canvas2.pack(side=BOTTOM,anchor=CENTER)


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

def changeLabel_1(var,checkVar0):
    var.set('等待训练')
    checkVar0 = int(checkVar0.get())
    if checkVar0 == 1:
        show_data()
    else:
        pass




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



def main(var,data,N_hours,N_train_hours):       #N_train_hours为训练数据量
    N_features = len(data[0])
    print('训练序列维度：',N_features,'\n')
    print('训练时间步长：',N_hours,'\n')
    print('训练集大小为：',N_train_hours,'\n')
    training(data,N_hours,N_features,N_train_hours)
    var.set('训练已完成')


def time_set():
    time.sleep(1)

def show_data():
    global file_path
    # load dataset
    dataset = read_csv(file_path, header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 4, 5, 6]
    i = 1
    i2 = 0
    # plot each column
    #pyplot.rcParams['font.sans-serif'] = ['KaiTi']
    #pyplot.rcParams['axes.unicode_minus'] = False
    fig=pyplot.figure()
    fig.canvas.set_window_title('数据集可视化（5s后自动退出）')
    # columns2=['rain','temperature','speed','NDVI','DEM','type','water']
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        # pyplot.title(columns2[i2], y=0.5, loc='right')
        #pyplot.rcParams['font.sans-serif'] = ['SimHei']
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
        i2 += 1
    pyplot.ion()
    pyplot.pause(4)
    pyplot.close()

    '''
    1.time 时间
    2.rain 降雨
    3.temperature 地表温度
    4.speed of wind 地表风速
    5.NDVI
    6.terrain 地形
    7.type of soil 土壤类型
    8.moisture content 含水率
    '''

if __name__ == '__main__':
    creat_windows()



