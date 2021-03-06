from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

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

def training(data,N_hours,N_features,N_train_hours):
    #load dataset
    values=data
    print(data[0],'开始调用')
    #scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # specify the number of lag hours
    n_hours = N_hours
    n_features = N_features
    print(n_hours,n_features)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    #reframed = series_to_supervised(values, n_hours, 1)
    print(reframed.shape)
    print(reframed.head(5))

    # split into train and test sets
    values = reframed.values
    #n_train_hours = 365 * 24
    n_train_hours = N_train_hours
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.ion()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    #print(yhat)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    print('output shape is ',yhat.shape)
    inv_yhat = concatenate((test_X[:, -n_features:-1],yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    print('predict result is\n',inv_yhat)

    # test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
    # # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:, 0]
    # # invert scaling for actual
    # test_y = test_y.reshape((len(test_y), 1))
    # inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:, 0]
    # # calculate RMSE
    # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)
    # pyplot.plot(inv_y, label='true')
    # pyplot.plot(inv_yhat, label='test')
    # pyplot.legend()
    # pyplot.show()

