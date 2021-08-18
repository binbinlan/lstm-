#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
# 设置常量
rnn_unit = 10         # 隐层神经元的个数
lstm_layers = 2       # 隐层层数
input_size = 7        # 输入神经元个数
output_size = 1       # 输出神经元个数（预测值）
lr=0.0006           # 学习率
# ——————————————————导入数据——————————————————————
f = open('dataset_2.csv')
df = pd.read_csv(f)     # 读入股票数据
data = df.iloc[:,2:10].values  # 取第3-10列 （2:10从2开始到9）
print(data)

# 获取训练集
# time_step 时间步，batch_size 每一批次训练多少个样例
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    batch_index = []
    data_train = data[train_begin:train_end]  # 前5800个数据作为训练集，后300个作为测试集
    normalized_train_data = (data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  # 标准化
    train_x, train_y = [], []   # 训练集
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size == 0:
           batch_index.append(i)
        x = normalized_train_data[i:i+time_step, :7]
        y = normalized_train_data[i:i+time_step, 7, np.newaxis]  # np.newaxis分别是在行或列上增加维度
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=5801):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data=(data_test-np.mean(data_test, axis=0))/np.std(data_test, axis=0)  # 标准化
    test_size = (len(normalized_test_data)+time_step-1)//time_step  # " // "表示整数除法。有size个sample
    print('size', test_size)
    test_x, test_y = [], []
    for i in range(test_size-1):
        x = normalized_test_data[i*time_step:(i+1)*time_step, :7]
        y = normalized_test_data[i*time_step:(i+1)*time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:, 7]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置、dropout参数
# weights:input weights+output weights
# 进入RNN的cell之前，要经过一层hidden layer
# cell计算完结果后再输出到output hidden layer
# 下面就定义cell前后的两层hidden layer，包括weights和biases

weights = {
         'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
         'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout 防止过拟合

# ——————————————————定义神经网络——————————————————
def lstmCell():
    # basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm
# 是的这里没用到它，不过只要改成return drop就可以加入dropout

def lstm(X):  # 参数：输入网络批次数目
    
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    # hidden layer for input to cell
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in)+b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    # cell
    # 包含多少个节点，forget_bias:初始的forget定义为1，也就是不忘记，state_is_tuple：
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    # RNN每次计算一次都会保留一个state
    # LSTM会保留两个state，lstm cell is divided into two parts(c_state,m_state),
    # 也就是主线的state(c_state),和分线的state(m_state)，会包含在元组（tuple）里边
    # state_is_tuple=True就是判定生成的是否为一个元组

    # 初始state,全部为0，慢慢的累加记忆
    init_state = cell.zero_state(batch_size,dtype=tf.float32)
    # outputs是一个list，每步的运算都会保存起来，
    # time_majortime的时间点是不是在维度为1的地方，我们的放在第二个维度，28steps
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    # hidden layer for outputs and final results
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states

# ————————————————训练模型————————————————————
# train_begin=2000
def train_lstm(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    print('pred,_',pred,_)
    # 损失函数
    # [-1]——列表从后往前数第一列，即pred为预测值，Y为真实值(Label)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    # 误差loss反向传播
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)


    with tf.Session() as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 重复训练10000次
        theloss = []
        for i in range(10):     # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step+1]], Y: train_y[batch_index[step]:batch_index[step+1]], keep_prob: 0.5})
            print("feed_dict", {X: train_x[batch_index[step]:batch_index[step+1]], Y: train_y[batch_index[step]:batch_index[step+1]], keep_prob: 0.5})

            print("Number of iterations:", i, " loss:", loss_)
            theloss.append(loss_)
        print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
        print("The train has finished")
    return theloss
theloss=train_lstm()

# ————————————————预测模型————————————————————
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    # 为什么要用tf.variable_scope来定义重复利用？
    # ——RNN会经常用到。
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y)*std[7]+mean[7]
        test_predict = np.array(test_predict)*std[7]+mean[7]
        # 相对误差=（测量值-计算值）/计算值×100%
        acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  # 偏差程度
        print("预测的相对误差:", acc)

        # 以折线图表示预测结果
        plt.figure(1)
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.xlabel('time value/day', fontsize=14)
        plt.ylabel('close value/point', fontsize=14)
        plt.title('predict-----blue,real-----red', fontsize=10)
        plt.show()

        print(theloss)
        plt.figure(2)
        plt.plot(list(range(len(theloss))), theloss, color='b', )
        plt.xlabel('times', fontsize=14)
        plt.ylabel('loss valuet', fontsize=14)
        plt.title('loss-----blue', fontsize=10)
        plt.show()

prediction()
