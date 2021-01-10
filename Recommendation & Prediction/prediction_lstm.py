from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
from numpy import split
import numpy as np
from numpy import array
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout

mpl.rcParams['figure.figsize'] = (20, 20)
long_update = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/2 - Latest/long_update.csv')
# long_update = long_update.rename(columns={"CRL": "crl"})
columns = [*long_update.columns]



class DataPrepMulti():
    def __init__(self, init_df, crl):
        self.init_df = init_df
        self.crl = crl
        self.n_out = 7
        self.time_steps = self.get_time_steps()
        self.n_features = len(['date', 'value', 'serie', 'cl'])

    def get_data(self):
        features = ['date', 'value', 'serie', 'cl']
        df = self.init_df[self.init_df['CRL'] == self.crl]

        df = df[features]
        df = df.set_index('date')
        df = df.groupby('date').sum()
        df = df.loc[:'2019-12-31']
        return(df)

    def get_scaled_data(self):
        scaler = MinMaxScaler((0,1))
        df = self.get_data()
        df_scaled = pd.DataFrame(scaler.fit_transform(df))
        df_scaled.index = df.index
        return(df_scaled, scaler)
    def get_scaled_y(self):
        scaler = MinMaxScaler((0,1))
        df = self.get_data()['value']
        df_scaled = pd.DataFrame(scaler.fit_transform(np.asarray(df).reshape(-1, 1)))
        df_scaled.index = df.index
        return(scaler)



    def get_time_steps(self):
        temp_df = self.get_data()
        time_steps = int(0.30 * len(temp_df))
        return(time_steps)

    def series_to_supervised(self, dropnan=True):
        data, test = self.get_scaled_data()
        n_in = self.time_steps
        n_out = self.n_out
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
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
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
        	agg.dropna(inplace=True)
        return agg

    def split_data(self):
        df = self.get_scaled_data()[0]
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
        self.train, self.test = train, test
        return (train, test)

    def split_data_not_scaled(self):
        df = self.get_data()
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
        self.train, self.test = train, test
        return (train, test)

    def to_supervised(self):
        time_steps = self.get_time_steps()
        train, test = self.split_data()
        train, test = np.array(train), np.array(test)
        data = train.reshape((train.shape[0]*train.shape[1], self.n_features))
        X, y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.time_steps
            out_end = in_end + self.n_out
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
    # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    def to_supersived_train_test(self):
        n_hours = self.time_steps
        reframed = self.series_to_supervised()
        values = reframed.values
        n_features = self.n_features
        n_train_hours = self.time_steps
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        n_obs = n_hours * n_features
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]
        print(train_X.shape, len(train_X), train_y.shape)
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        return(train_X, train_y, test_X, test_y)

    def fit_multi_model(self):
        train_X, train_y, test_X, test_y = self.to_supersived_train_test()
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.fit(train_X, train_y, epochs=30, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        return(model)
# fit network
    # def new_split_data(self):
    #     reframed = self.series_to_supervised(self.get_scaled_data(), self.time_steps, self.n_out)
    #     values = reframed.values
    #     n_train_hours = 365 * 24
    #     train = values[:n_train_hours, :]
    #     test = values[n_train_hours:, :]
    #     # split into input and outputs
    #     n_obs = n_hours * n_features
    #     train_X, train_y = train[:, :n_obs], train[:, -n_features]
    #     test_X, test_y = test[:, :n_obs], test[:, -n_features]
    #     print(train_X.shape, len(train_X), train_y.shape)
    #     # reshape input to be 3D [samples, timesteps, features]
    #     train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    #     test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    #     print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


class_me = DataPrepMulti(long_update, 'CRL5000017972')
data = class_me.get_data()
plt.plot(data)
scaler = class_me.get_scaled_data()[1]
scaler_2 = class_me.get_scaled_y()
train_X, train_y, test_X, test_y = class_me.to_supersived_train_test()
model = class_me.fit_multi_model()

feature = class_me.n_features
time = class_me.time_steps
# yhat = scaler.inverse_transform(yhat)
yhat = model.predict(test_X)
yhat

test_X = test_X.reshape((test_X.shape[0], time*feature))
test_X = pd.DataFrame(test_X)
# invert scaling for forecast
inv_yhat = pd.concat((yhat, test_X[:, -7:]), axis=1)
inv_yhat = scaler_2.inverse_transform(yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
train_X, train_y, test_X, test_y = class_me.to_supersived_train_test()
train_X, train_y, test_X, test_y
train, test_2 = test.split_data()
# test.get_data()
test.to_supervised()


time_weekly = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/1 - Python_Files/Cleaning/final_csv_weekly.csv')
time_daily = pd.read_csv('/Users/Julien/Desktop/LBS/Cours/8 - TERM 3/London Lab/Data.nosync/1 - Python_Files/Cleaning/long_update.csv')
time_weekly = time_weekly.dropna()

time_daily = time_daily.dropna()
time_daily
# time_daily.index = pd.to_datetime(time_daily.index)

time_daily
time_weekly = time_weekly.dropna()
count_df = time_daily.groupby('CRL').count().sort_values(by = 'date', ascending = False)
print(count_df)

class DataPrep():
    def __init__(self, init_df, crl):
        self.init_df = init_df
        self.crl = crl
        self.n_out = 7
        self.time_steps = self.get_time_steps()

    def get_data(self):
        features = ['date', 'value']
        df = self.init_df[self.init_df['CRL'] == self.crl]

        df = df[features]
        df = df.set_index('date')
        df = df.groupby('date').sum()
        df = df.loc[:'2019-12-31']
        return(df)

    def get_scaled_data(self):
        scaler = MinMaxScaler((0,1))
        df = self.get_data()
        df_scaled = pd.DataFrame(scaler.fit_transform(df))
        df_scaled.index = df.index
        return(df_scaled, scaler)


    def get_time_steps(self):
        temp_df = self.get_data()
        time_steps = int(0.11 * len(temp_df))
        return(time_steps)

    def split_data(self):
        df = self.get_scaled_data()[0]
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
        self.train, self.test = train, test
        return (train, test)

    def split_data_not_scaled(self):
        df = self.get_data()
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
        self.train, self.test = train, test
        return (train, test)


    def to_supervised(self):
        time_steps = self.get_time_steps()
        train, test = self.split_data()
        train, test = np.array(train), np.array(test)
        data = train.reshape((train.shape[0]*train.shape[1], 1))
        X, y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.time_steps
            out_end = in_end + self.n_out
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
    # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    def to_supervised_simple(self):
        time_steps = self.get_time_steps()
        train, test = self.split_data()
        train, test = np.array(train), np.array(test)
        data = train.reshape((train.shape[0]*train.shape[1], 1))
        X, y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.time_steps
            out_end = in_end + 1
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
    # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    def create_dataset(self):
        look_back = self.time_steps
        dataset, test = self.split_data()
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)


    # def to_supervised_test(self):
    #     time_steps = self.get_time_steps()
    #     train, test = self.split_data()
    #     train, test = np.array(train), np.array(test)
    #     data = test.reshape((test.shape[0]*test.shape[1], 1))
    #     X, y = list(), list()
    #     in_start = 0
    #     for _ in range(len(data)):
    #         in_end = in_start + self.time_steps
    #         out_end = in_end + self.n_out
    #         if out_end <= len(data):
    #             x_input = data[in_start:in_end, 0]
    #             x_input = x_input.reshape((len(x_input), 1))
    #             X.append(x_input)
    #             y.append(data[in_end:out_end, 0])
    # # move along one time step
    #         in_start += 1
    #     return np.array(X), np.array(y)





my_class = DataPrep(time_weekly, 'CRL5000015752')
data, scaler = my_class.get_scaled_data()
data.index = pd.to_datetime(data.index)
plt.plot(data)

# train, test = my_class.to_supervised_simple()
#
# # my_class.time_steps
# train, test = my_class.split_data()
# look_back = my_class.time_steps
# train[0:look_back, 0]
# # train, test = my_class.create_dataset()
# X, Y = [], []
# for i in range(len(train)-look_back-1):
#     print(i)
#     a = train[i:(i+look_back), 0]
#     X.append(a)
#     Y.append(train[i + look_back, 0])


# x,y = my_class.to_supervised_test()
# data

# plt.plot(my_class.get_data())




class Model():
    def __init__(self, data_object):
        self.df = data_object.get_data()
        self.train, self.test = data_object.split_data()
        self.time_steps = data_object.get_time_steps()
        self.data_object = data_object

    def build_model_v2(self):
        train_x, train_y = self.data_object.to_supervised()
        verbose = 1
        epochs = 70
        batch_size = 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs))
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)
        return model

    def get_multi_step_prediction(self):
        time_steps = self.time_steps
        train, test = self.data_object.split_data()
        model = self.build_model_v2()
        scaler = self.data_object.get_scaled_data()[1]
        input_x = np.array(train[-time_steps:])
        input_x = input_x.reshape((1, len(input_x), 1))
        yhat = model.predict(input_x, verbose=1)
        predictions = scaler.inverse_transform(yhat)
        final = train.iloc[len(train)-1]

        final_date = pd.to_datetime(final.name)

        next_date_daily = final_date +  datetime.timedelta(days=1)
        next_date_weekly = final_date +  datetime.timedelta(days=1)
        dates = [final_date]
        for x in range (len(predictions[0])):
            new_date = dates[-1] + datetime.timedelta(days=1)
            dates.append(new_date)

        # dates = [final_date + datetime.timedelta(days=1) for x in range(len(predictions))]
        return(predictions, dates)

    def plot_predictions(self):
        predictions, dates = self.get_multi_step_prediction()
        dates_final = [*dates[1:]]
        predictions_final = [*predictions[0]]

        df = pd.DataFrame(list(zip(dates_final, predictions_final)), columns =['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])

        df = df.set_index('date')

        train, test = self.data_object.split_data_not_scaled()
        df_concat = pd.concat([train, df])
        df_concat = df_concat.reset_index()
        df_concat['value'] = to_numeric(df_concat['value'])
        df_concat['date'] = pd.to_datetime(df_concat['date'])
        df_concat = df_concat.set_index('date')
        final = train.iloc[len(train)-1]
        final_date = pd.to_datetime(final.name)
        final_value = final.value
        # plt.plot(final_date, final_value, color = 'red', marker = 'o')
        # plt.plot(df_concat)
        # plt.show()

        # fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

        plt.plot(final_date, final_value, color = 'red', marker = 'o', markersize = 10)
        plt.plot(df_concat, color = (18.37/100,51.02/100,30.61/100))
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.savefig('predictions.png')
        # f, ax = plt.subplots()
        # ax.plot(df_concat)
        # ax.plot(final_date, final_value, color = 'red', marker = 'o')
        # return(ax)



    def build_simple_model(self):
        X_train, Y_train = self.data_object.to_supervised_simple()
        model = Sequential()
        model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = model.fit(X_train, Y_train, epochs=30, batch_size=1)
        return(model)


    def plot_train_test_model(self):
        train_x, train_y = self.data_object.to_supervised_simple()
        # test_x, test_y = self.data_object.to_supervised_test()
        model = self.build_simple_model()
        predictions_train = model.predict(train_x)
        # predictions_test = model.predict(test_x)
        scaler = self.data_object.get_scaled_data()[1]
        predictions_train = scaler.inverse_transform(predictions_train)



        train, test = self.data_object.split_data_not_scaled()
        predictions_train = pd.DataFrame(predictions_train)
        predictions_train.index = pd.to_datetime(train.index[self.time_steps:])
        train.index = pd.to_datetime(train.index)
        # predictions = scaler.inverse_transform(predictions_train)
        plt.plot(train, color = (18.37/100,51.02/100,30.61/100))
        plt.plot(predictions_train, color = 'red', alpha = 0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.savefig('train_vs_test.png')
        # return(predictions_train)


my_next_class = Model(my_class)
my_next_class.plot_predictions()

my_next_class.plot_train_test_model()
# predictions, dates = my_next_class.get_multi_step_prediction()
# plot_pred = my_next_class.plot_predictions()
# plot_pred.get_figure()
# plt.show()

# my_next_class.plot_train_test_model()
predictions_train
# dates_final = [*dates[1:]]
# predictions_final = [*predictions[0]]
#
# df = pd.DataFrame(list(zip(dates_final, predictions_final)), columns =['date', 'value'])
# df['date'] = pd.to_datetime(df['date'])
#
# df = df.set_index('date')
#
#
# train
# df_concat = pd.concat([train, df])
# df_concat = df_concat.reset_index()
# df_concat['value'] = to_numeric(df_concat['value'])
# df_concat['date'] = pd.to_datetime(df_concat['date'])
# df_concat = df_concat.set_index('date')
# final = train.iloc[len(train)-1]
# print(final)
#
# final_date = pd.to_datetime(final.name)
# final_value = final.value
# plt.plot(final_date, final_value, color = 'red', marker = 'o')
# plt.plot(df_concat)
# plt.show()
# def run_multi_step_model(crl = crl):
#     scaler = MinMaxScaler((0,1))
#     df = get_scaled_data(crl)
#     time_steps = int(0.11 * len(df))
#     # df = pd.DataFrame(scaler.fit_transform(df))
#     train, test = split_data(df)
#     train = np.array(train)
#     train_x, train_y = to_supervised(train, time_steps)
#     new_model = build_model_v2(train_arr, time_steps)
#     return(time_steps, scaler, train, new_model)
