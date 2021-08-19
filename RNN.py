import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from pandas import DataFrame
from math import sqrt,nan


plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

data = pd.read_excel(r'sheet.xls',index_col=None,header=0)
data = data.values

print(data)

# 生成时间序列
time_steps = int(input('Please input timestep:'))

# 此函数用于生成时间序列，其时间步长可任意调整=time_steps
def gen_series_data(timesteps, input_data):
    print(input_data.shape)
    new_data = np.zeros((input_data.shape[0]-timesteps, input_data.shape[1]*timesteps+1))
    print(new_data.shape)
    for i in range(new_data.shape[0]):
        #print(i)
        for j in range(new_data.shape[1]):
            #print(j)
            if j < input_data.shape[1]:
                new_data[i, j] = input_data[i, j]
            elif j >= input_data.shape[1] and j != new_data.shape[1]-1:
                new_data[i, j] = input_data[i+(int(j/input_data.shape[1])),
                                            j%input_data.shape[1]]
            elif j == new_data.shape[1]-1:
                new_data[i, j] = input_data[i+(int(j/input_data.shape[1])),
                                            -1]
    return new_data

series_data = gen_series_data(time_steps, data)
print(series_data)
print(series_data.shape)
# 取最后一周做测试集
train_data_rows = int(series_data.shape[0]*0.8)
test_data_rows = series_data.shape[0]-train_data_rows
cols = series_data.shape[1]

train_data = np.zeros((train_data_rows, cols))
test_data = np.zeros((test_data_rows, cols))

for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        train_data[i, j] = series_data[i, j]

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        test_data[i, j] = series_data[i+train_data_rows, j]

# 分解输入与输出
train_x = train_data[:, :-1]
train_y = train_data[:, -1]
train_y = train_y.reshape(-1, 1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]
test_y = test_y.reshape(-1, 1)

real_values = np.zeros((test_y.shape[0], test_y.shape[1]))
for i in range(real_values.shape[0]):
    real_values[i, 0] = test_y[i, 0]

real_value = np.zeros((series_data.shape[0], test_y.shape[1]))
for i in range(real_value.shape[0]):
    if i < train_y.shape[0]:
        real_value[i, 0] = train_y[i, 0]
    else:
        real_value[i, 0] = test_y[i-train_y.shape[0], 0]
print(real_value)
print(real_value.shape)
# 定义输入的归一化方法
def norm_x(input_data):
    max_value = list()
    min_value = list()
    for j in range(input_data.shape[1]):
        max_value.append(max(input_data[:, j]))
        min_value.append(min(input_data[:, j]))
    print(max_value)
    print(min_value)
    for j in range(input_data.shape[1]):
        for i in range(input_data.shape[0]):
            if (max_value[j]-min_value[j]) == 0:
                print(j)
                print(max_value[j])
                print(min_value[j])
                print(1)
            input_data[i, j] = round((input_data[i, j] - min_value[j])/(max_value[j]-min_value[j]),3)
    return  input_data

# 定义输出的归一化方法，此时需返回最大值与最小值以备反归一化
def norm_y(input_data):
    min_value = min(input_data[:, 0])
    max_value = max(input_data[:, 0])
    for i in range(input_data.shape[0]):
        input_data[i, 0] = (input_data[i, 0]-min_value)/(max_value-min_value)
    return input_data, max_value, min_value

# 生成归一化的序列数据
norm_train_x = norm_x(train_x)
norm_test_x = norm_x(test_x)
norm_train_y, max_train_y, min_train_y = norm_y(train_y)
norm_test_y, max_test_y, min_test_y = norm_y(test_y)

# 搭建模型与训练模型方法
def build_model(neurons,input_x, input_y, epochs, batch_size, learning_rate=0.001,verbose=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_x.shape[1], activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(neurons*2, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    history = model.fit(input_x, input_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)
    plt.plot(history.history['loss'], c='k', label='loss')
    plt.plot(history.history['val_loss'], c='r',label='val_loss')
    plt.legend()
    plt.show()
    return model

model_lstm = build_model(80, norm_train_x, norm_train_y, epochs=300, batch_size=1)
pred_values = model_lstm.predict(norm_test_x)
pred_values = pred_values.reshape(-1, 1)

pred = np.zeros((pred_values.shape[0], pred_values.shape[1]))
# 反归一化
for i in range(pred_values.shape[0]):
    pred[i, 0] = pred_values[i, 0]*(max_test_y-min_test_y)+min_test_y

# 定义计算误差MAPE的方法
def cal_MAPE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += (abs(pred[i, 0]-true[i, 0])/true[i, 0])
    error = error / pred.shape[0]
    return error
# 计算RMSE
def cal_RMSE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += (pred[i, 0]-true[i, 0])**2
    error = error / pred.shape[0]
    error = sqrt(error)
    return error
# 计算MAE
def cal_MAE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += abs(pred[i, 0]- true[i, 0])
    error = error/pred.shape[0]
    return error

real_pred = np.zeros((real_value.shape[0], pred.shape[1]))
for i in range(real_pred.shape[0]):
    if i < train_y.shape[0]:
        real_pred[i, 0] = nan
    else:
        real_pred[i, 0] = pred[i-train_y.shape[0], 0]

MAPE = cal_MAPE(pred, real_values)
MAE = cal_MAE(pred, real_values)
RMSE = cal_RMSE(pred, real_values)
print("MAPE is: "+str(MAPE*100)+'%')
print("MAE is: "+str(MAE))
print("RMSE is: "+str(RMSE))
# 作图
plt.plot(real_value, c='k', label='true')
plt.plot(real_pred, c='r',label='pred')
plt.legend()
plt.show()

# 输出预测结果
pred = DataFrame(pred)
pred.to_csv(r'pred.csv', header=None, index=None)

model_lstm.save(r'model.h5')