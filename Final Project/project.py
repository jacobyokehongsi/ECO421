# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:56:22 2021

@author: Yuhang
"""
from math import floor
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

def load_data():
    appl = pd.read_csv('AAPL.csv', parse_dates=['Date'])
    tsla = pd.read_csv('TSLA.csv', parse_dates=['Date'])
    amzn = pd.read_csv('AMZN.csv', parse_dates=['Date'])
    us_covid = pd.read_csv('us_covid_index.csv', parse_dates=['Day'])
    us_covid = us_covid[['Day', 'stringency_index']]
    us_covid = us_covid.rename(columns={'Day': 'Date', 'stringency_index': 'strin_us'})
    vix = pd.read_csv('vix_index.csv', parse_dates=['Date'])
    vix = vix[['Date', 'VIX']]

    return appl, tsla, amzn, us_covid, vix
# take the latter 40% of available data for validation of investment return 
# 0.04  0.01 0.01 0.01 0.04 0.05 0.03 0.05 0.02 0.04
# find the predicted price on these days (index 0, 20, to 180), if bigger than
# actual price of last day, invest in stock, otherwise, invest in t-bill and 
# hold until maturity. 
# 3 strategies: switch, hold t-bill and hold stock only. 
# compare the return at the end. excpet for tbill only, assume inintial investment of stock.
#fucntion below calcuates the rsi index for current period.
#Moving average
#is a popular and simple indicator for trends.RSI is simple a
#indicator which help traders identify turning points.
# trend and turning point of prices. 
# simple moving average is the moving average of a certain period, where 
# expotential moving average assigns more weight to the most recent prices.


def add_rsi_ema(data):
    data1 = data['Adj Close']
    data_d = data1.diff()

    data_d = data_d[1:]

    window_length = 10

    up, down = data_d.clip(lower=0), data_d.clip(upper=0).abs()

    # Calculate the RSI based on EWMA
    # Reminder: Try to provide at least `window_length * 4` data points!
    roll_up = up.ewm(span=window_length).mean()
    roll_down = down.ewm(span=window_length).mean()
    rs = roll_up / roll_down
    rsi_ewma = 100.0 - (100.0 / (1.0 + rs))
    # rs can't fall below zero. it measures the relative upward movement to down movement.
    data['RSI'] = rsi_ewma
    data2 = data[['Date', 'Adj Close', 'RSI']]
    data2 = data2[1:]
    data2.ta.ema(close='Adj Close', length=5, append=True)
    return data2


def merging(dataset_rsi, cov, v):
    merged = dataset_rsi.merge(v, on='Date')
    # merged = merged.merge(china_covid,on='Date')
    merged = merged.merge(cov, on='Date')

    a = merged['Adj Close'].pct_change(1)
    merged['return'] = a
    b = merged['Adj Close'].shift(1)
    merged['price_lag'] = b
    c = merged['return'].shift(1)
    merged['return_lag'] = c
    merged['pos_neg'] = (merged['return'] > 0)
    merged['pos_neg'] = merged['pos_neg'].astype(int)
    d = merged['VIX'].shift(1)
    merged['VIX_lag'] = d
    e = merged['strin_us'].shift(1)
    merged['us_lag'] = e
    f = merged['RSI'].shift(1)
    merged['rsi_lag'] = f
    g = merged['pos_neg'].shift(1)
    merged['pn_lag'] = g
    h = merged['EMA_5'].shift(1)
    merged['EMA_lag'] = h
    return merged


def predict(company_merged, starting_index):
    data2 = company_merged[
        ['Date', 'Adj Close', 'return', 'return_lag', 'pos_neg', 'VIX_lag', 'us_lag', 'rsi_lag', 'pn_lag', 'EMA_lag']]
    comp_len = floor(len(company_merged)/5)
    train2 = data2[starting_index:comp_len+1]
    print(train2)
    SS = StandardScaler()
    train2 = SS.fit_transform(train2)
    # train_X = train[['return_lag','rsi_lag']]
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    train_X2 = train_X2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']].to_numpy()
    train_Y2 = train2[['Adj Close']]
    train_Y2 = train_Y2['Adj Close'].to_numpy()
    test2 = data2[comp_len+1:]
    test2 = SS.fit_transform(test2)
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    test_X2 = test_X2.to_numpy()
    test_Y2 = test2[['Adj Close']]
    test_Y2 = test_Y2['Adj Close'].to_numpy()
    list = []
    list1 = []
    innlist1 = []
    innlist2 = []
    for i in range(0, 20):
        clf2 = MLPRegressor(max_iter=2000)
        clf2.fit(train_X2, train_Y2)
        Y_predict2 = clf2.predict(test_X2)
        innlist1.append(mean_squared_error(test_Y2, Y_predict2))
    # accuracy_score(test_Y,clf.predict(test_X))
    for i in range(0, 20):
        dec = DecisionTreeRegressor()
        dec.fit(train_X2, train_Y2)
        Y_predict2 = dec.predict(test_X2)
        innlist2.append(mean_squared_error(test_Y2, Y_predict2))
    list.append(sum(innlist1) / 20)
    list1.append(sum(innlist2) / 20)

    # manually split test and train
    # time series validation.
    # 93  93*2 93*3 93*4
    train2 = data2[starting_index:comp_len * 2 + 1]
    # train_X = train[['return_lag','rsi_lag']]
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    train_X2 = train_X2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']].to_numpy()
    train_Y2 = train2[['Adj Close']]
    train_Y2 = train_Y2['Adj Close'].to_numpy()
    test2 = data2[comp_len * 2 + 1:]
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    test_X2 = test_X2.to_numpy()
    test_Y2 = test2[['Adj Close']]
    test_Y2 = test_Y2['Adj Close'].to_numpy()
    innlist1 = []
    innlist2 = []
    for i in range(0, 20):
        clf2 = MLPRegressor(max_iter=2000)
        clf2.fit(train_X2, train_Y2)
        Y_predict2 = clf2.predict(test_X2)
        innlist1.append(mean_squared_error(test_Y2, Y_predict2))
    # accuracy_score(test_Y,clf.predict(test_X))
    for i in range(0, 20):
        dec = DecisionTreeRegressor()
        dec.fit(train_X2, train_Y2)
        Y_predict2 = dec.predict(test_X2)
        innlist2.append(mean_squared_error(test_Y2, Y_predict2))
    list.append(sum(innlist1) / 20)
    list1.append(sum(innlist2) / 20)

    train2 = data2[5:comp_len * 3 + 1]
    # train_X = train[['return_lag','rsi_lag']]
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    train_X2 = train_X2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']].to_numpy()
    train_Y2 = train2[['Adj Close']]
    train_Y2 = train_Y2['Adj Close'].to_numpy()
    test2 = data2[comp_len * 3 + 1:]
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    test_X2 = test_X2.to_numpy()
    test_Y2 = test2[['Adj Close']]
    test_Y2 = test_Y2['Adj Close'].to_numpy()
    innlist1 = []
    innlist2 = []
    for i in range(0, 20):
        clf2 = MLPRegressor(max_iter=2000)
        clf2.fit(train_X2, train_Y2)
        Y_predict2 = clf2.predict(test_X2)
        innlist1.append(mean_squared_error(test_Y2, Y_predict2))
    # accuracy_score(test_Y,clf.predict(test_X))
    for i in range(0, 20):
        dec = DecisionTreeRegressor()
        dec.fit(train_X2, train_Y2)
        Y_predict2 = dec.predict(test_X2)
        innlist2.append(mean_squared_error(test_Y2, Y_predict2))
    list.append(sum(innlist1) / 20)
    list1.append(sum(innlist2) / 20)

    train2 = data2[starting_index:comp_len * 4 + 1]
    # train_X = train[['return_lag','rsi_lag']]
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    train_X2 = train_X2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']].to_numpy()
    train_Y2 = train2[['Adj Close']]
    train_Y2 = train_Y2['Adj Close'].to_numpy()
    test2 = data2[comp_len * 4 + 1:]
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    test_X2 = test_X2.to_numpy()
    test_Y2 = test2[['Adj Close']]
    test_Y2 = test_Y2['Adj Close'].to_numpy()
    innlist1 = []
    innlist2 = []
    for i in range(0, 20):
        clf2 = MLPRegressor(max_iter=2000, hidden_layer_sizes=(100, 50), learning_rate=0.5)
        clf2.fit(train_X2, train_Y2)
        Y_predict2 = clf2.predict(test_X2)
        innlist1.append(mean_squared_error(test_Y2, Y_predict2))
    # accuracy_score(test_Y,clf.predict(test_X))
    for i in range(0, 20):
        dec = DecisionTreeRegressor()
        dec.fit(train_X2, train_Y2)
        Y_predict2 = dec.predict(test_X2)
        innlist2.append(mean_squared_error(test_Y2, Y_predict2))
    list.append(sum(innlist1) / 20)
    list1.append(sum(innlist2) / 20)
    final1 = sum(list) / 4
    final2 = sum(list1) / 4

    return final1, final2


def main():
    appl, tsla, amzn, us_covid, vix = load_data()

    tsla_rsi = add_rsi_ema(tsla)
    appl_rsi = add_rsi_ema(appl)  # rsi index with expotential weighted moving average
    amzn_rsi = add_rsi_ema(amzn)

    appl_merged = merging(appl_rsi, us_covid, vix)
    appl_merged.plot('Date', 'Adj Close')
    plt.ylabel('Apple Stock Price, USD')
    appl_merged.plot('Date', 'return')
    plt.ylabel('Apple Stock Return, fraction')

    tsla_merged = merging(tsla_rsi, us_covid, vix)
    tsla_merged.plot('Date', 'Adj Close')
    plt.ylabel('Tesla Stock Price, USD')
    tsla_merged.plot('Date', 'return')
    plt.ylabel('Tesla Stock Return, fraction')

    amzn_merged = merging(amzn_rsi, us_covid, vix)
    amzn_merged.plot('Date', 'Adj Close')
    plt.ylabel('Amazon Stock Price, USD')
    amzn_merged.plot('Date', 'return')
    plt.ylabel('Amazon Stock Return, fraction')

    fin1, fin2 = predict(appl_merged, 5)
    print(fin1, fin2)


if __name__ == "__main__":
    main()

# hey jacob, this is a very imcomplete version of stock prediction: I only used 
# apple stock as a starting example. I chose the covid period (2020/1/21 to present)
# as it might add more motivation to our report. (it will not be hard to use periods before that)
# heres the variables i inlcuded for predicting
# stock price directly: vix index (the one prof talks about last week), rsi(relative
# strength index) and the ema index, both are calculated using stock price to capture past
# stock momentums. aside from these three, i included  covid stringency index
# of us and china from https://ourworldindata.org/grapher/covid-stringency-index?time=2020-08-27
# to reflect the covid crisis impact on internationally operated big tech firms like 
# apple and amazon and so on. i think firm financial ratios are two infrequent for day to day
# forecast. till now, the result is best for direct stock price prediction, and it is not 
# that good for both return level and return direction prediction (close two 5% addition to the overall prediction accuracy for direction prediction).
