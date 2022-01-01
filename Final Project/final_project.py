from math import floor
import numpy as np
import pandas as pd
import pandas_ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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


def predict_level_reg(company_merged, starting_index, num_split, classifier):
    data2 = company_merged[
        ['Date', 'Adj Close', 'return', 'return_lag', 'pos_neg', 'VIX_lag', 'us_lag', 'rsi_lag', 'pn_lag', 'EMA_lag']]
    comp_len = floor(len(company_merged)/5)
    train2 = data2[starting_index:comp_len * num_split + 1]
    # SS = StandardScaler()
    # print(train2)
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    # train_X2 = SS.fit_transform(train_X2)
    # train_X2 = SS.fit_transform(train_X2)
    train_Y2 = train2[['Adj Close']]
    train_Y2 = train_Y2['Adj Close'].to_numpy()
    test2 = data2[comp_len * num_split + 1:comp_len * (num_split + 1) + 1]
    # testing window of length 93
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'return_lag', 'us_lag']]
    # test_X2 = SS.fit_transform(test_X2)
    test_Y2 = test2[['Adj Close']]
    test_Y2 = test_Y2['Adj Close'].to_numpy()

    if classifier == 'mlp':
        clf2 = MLPRegressor(max_iter=250000)
        # clf2 = MLPRegressor(solver='lbfgs', max_iter=250000)
        clf2.fit(train_X2, train_Y2)
        y_pred_mlp = clf2.predict(test_X2)

        plt.figure()
        plt.plot(y_pred_mlp)
        plt.plot(test_Y2)
        plt.title('Predicted and Actual values of Adj Close using Multilayer Perceptron (NN)')
        plt.xlabel('Days since 2020-07-15')
        plt.ylabel('Adj Close')
        plt.legend(['Predicted', 'Actual'])

        mse = mean_squared_error(test_Y2, y_pred_mlp)
        # print("MLP Mean Squared Error:", mse)
        return mse

    elif classifier == 'dec_tree':
        dec = DecisionTreeRegressor()
        dec.fit(train_X2, train_Y2)
        y_pred_dec = dec.predict(test_X2)

        plt.figure()
        plt.plot(y_pred_dec)
        plt.plot(test_Y2)
        plt.title('Predicted and Actual values of Adj Close using Decision Trees')
        plt.xlabel('Days since 2020-07-15')
        plt.ylabel('Adj Close')
        plt.legend(['Predicted', 'Actual'])

        mse = mean_squared_error(test_Y2, y_pred_dec)
        # print("Decision Tree Mean Squared Error:", mse)

        return mse

    elif classifier == 'rf':
        rf = RandomForestRegressor()
        rf.fit(train_X2, train_Y2)
        y_pred_rf = rf.predict(test_X2)

        plt.figure()
        plt.plot(y_pred_rf)
        plt.plot(test_Y2)
        plt.title('Predicted and Actual values of Adj Close using Random Forests')
        plt.xlabel('Days since 2020-07-15')
        plt.ylabel('Adj Close')
        plt.legend(['Predicted', 'Actual'])

        mse = mean_squared_error(test_Y2, y_pred_rf)
        # print("Random Forest Mean Squared Error:", mse)

        return mse

    elif classifier == 'lin_reg':
        lr = LinearRegression()
        lr.fit(train_X2, train_Y2)
        y_pred_lr = lr.predict(test_X2)

        plt.figure()
        plt.plot(y_pred_lr)
        plt.plot(test_Y2)
        plt.title('Predicted and Actual values of Adj Close using Linear Regression')
        plt.xlabel('Days since 2020-07-15')
        plt.ylabel('Adj Close')
        plt.legend(['Predicted', 'Actual'])

        mse = mean_squared_error(test_Y2, y_pred_lr)
        # print("Linear Regression Mean Squared Error:", mse)

        return mse


def predict_level_class(company_merged, starting_index, num_split, classifier):
    data2 = company_merged[
        ['Date', 'Adj Close', 'return', 'price_lag', 'pos_neg', 'VIX_lag', 'us_lag', 'rsi_lag', 'pn_lag', 'EMA_lag']]
    comp_len = floor(len(company_merged)/5)
    train2 = data2[starting_index:comp_len * num_split + 1]
    # print(train2)
    train_X2 = train2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'price_lag', 'us_lag']]
    SS = StandardScaler()
    train_X2 = SS.fit_transform(train_X2)
    train_Y2 = train2[['pos_neg']]
    train_Y2 = train_Y2['pos_neg']
    test2 = data2[comp_len * num_split + 1:comp_len * (num_split + 1) + 1]
    # testing window of length 93
    # test_X = test[[,'return_lag','rsi_lag']]
    test_X2 = test2[['rsi_lag', 'VIX_lag', 'EMA_lag', 'price_lag', 'us_lag']]
    test_X2 = SS.fit_transform(test_X2)
    test_Y2 = test2[['pos_neg']]
    test_Y2 = test_Y2['pos_neg']

    if classifier == 'mlp':
        # clf2 = MLPClassifier(solver='lbfgs', max_iter=250000)
        clf2 = MLPClassifier(max_iter=250000)
        clf2.fit(train_X2, train_Y2)
        y_pred_mlp = clf2.predict(test_X2)

        acc = accuracy_score(test_Y2, y_pred_mlp)
        # print("Accuracy MLP:", acc)
        return acc

    elif classifier == 'dec_tree':
        dec = DecisionTreeClassifier()
        dec.fit(train_X2, train_Y2)
        y_pred_dec = dec.predict(test_X2)

        acc = accuracy_score(test_Y2, y_pred_dec)
        # print("Accuracy Decision Tree:", acc)
        return acc

    elif classifier == 'logit':
        logit = LogisticRegression()
        logit.fit(train_X2, train_Y2)
        y_pred_logit = logit.predict(test_X2)

        acc = accuracy_score(test_Y2, y_pred_logit)
        # print("Accuracy Logit:", acc)
        return acc

    elif classifier == 'rf':
        rf = RandomForestClassifier()
        rf.fit(train_X2, train_Y2)
        y_pred_rf = rf.predict(test_X2)

        acc = accuracy_score(test_Y2, y_pred_rf)
        # print("Accuracy Random Forest:", acc)
        return acc

    return


def main():
    np.random.seed(50)
    appl, tsla, amzn, us_covid, vix = load_data()

    tsla_rsi = add_rsi_ema(tsla)
    appl_rsi = add_rsi_ema(appl)  # rsi index with expotential weighted moving average
    amzn_rsi = add_rsi_ema(amzn)

    appl_merged = merging(appl_rsi, us_covid, vix)
    # appl_merged.plot('Date', 'Adj Close')
    # plt.ylabel('Apple Stock Price, USD')
    # appl_merged.plot('Date', 'return')
    # plt.ylabel('Apple Stock Return, fraction')

    tsla_merged = merging(tsla_rsi, us_covid, vix)
    # tsla_merged.plot('Date', 'Adj Close')
    # plt.ylabel('Tesla Stock Price, USD')
    # tsla_merged.plot('Date', 'return')
    # plt.ylabel('Tesla Stock Return, fraction')

    amzn_merged = merging(amzn_rsi, us_covid, vix)
    # amzn_merged.plot('Date', 'Adj Close')
    # plt.ylabel('Amazon Stock Price, USD')
    # amzn_merged.plot('Date', 'return')
    # plt.ylabel('Amazon Stock Return, fraction')

    # predict_level_reg(company_merged, starting_index, num_split, classifier)
    comp_dict = {'AAPL': appl_merged, 'TSLA': tsla_merged, 'AMZN': amzn_merged}

    # predict_level_reg(appl_merged, 5, 4, 'lin_reg')  # plotting predicted returns with best model
    # predict_level_reg(tsla_merged, 5, 4, 'lin_reg')  # plotting predicted returns with best model
    # predict_level_reg(amzn_merged, 5, 4, 'mlp')  # plotting predicted returns with best model

    for name, c in comp_dict.items():
        mlp_mse_lst = []
        for i in range(1, 5):
            mlp_mse = predict_level_reg(c, 5, i, 'mlp')
            mlp_mse_lst.append(mlp_mse)
        print("{} mlp_mse_lst avg. mse:".format(name), np.mean(mlp_mse_lst))

        lin_reg_mse_lst = []
        for i in range(1, 5):
            lin_reg_mse = predict_level_reg(c, 5, i, 'lin_reg')
            lin_reg_mse_lst.append(lin_reg_mse)
        print("{} lin_reg_mse_lst avg. mse:".format(name), np.mean(lin_reg_mse_lst))

        dec_tree_mse_lst = []
        for i in range(1, 5):
            dec_tree_mse = predict_level_reg(c, 5, i, 'dec_tree')
            dec_tree_mse_lst.append(dec_tree_mse)
        print("{} dec_tree_mse_lst avg. mse:".format(name), np.mean(dec_tree_mse_lst))

        rf_mse_lst = []
        for i in range(1, 5):
            rf_mse = predict_level_reg(c, 5, i, 'rf')
            rf_mse_lst.append(rf_mse)
        print("{} rf_mse_lst avg. mse:".format(name), np.mean(rf_mse_lst))

        print("#################################################")

        mlp_acc_lst = []
        for i in range(1, 5):
            mlp_acc = predict_level_class(c, 5, i, 'mlp')
            mlp_acc_lst.append(mlp_acc)
        print("{} mlp_acc_lst avg. acc:".format(name), np.mean(mlp_acc_lst))

        logit_acc_lst = []
        for i in range(1, 5):
            logit_acc = predict_level_class(c, 5, i, 'logit')
            logit_acc_lst.append(logit_acc)
        print("{} logit_acc_lst avg. acc:".format(name), np.mean(logit_acc_lst))

        dec_tree_acc_lst = []
        for i in range(1, 5):
            dec_tree_acc = predict_level_class(c, 5, i, 'dec_tree')
            dec_tree_acc_lst.append(dec_tree_acc)
        print("{} dec_tree_acc_lst avg. acc:".format(name), np.mean(dec_tree_acc_lst))

        rf_acc_lst = []
        for i in range(1, 5):
            rf_acc = predict_level_class(appl_merged, 5, i, 'rf')
            rf_acc_lst.append(rf_acc)
        print("{} rf_acc_lst avg. acc:".format(name), np.mean(rf_acc_lst))

        print("#################################################")

    # plt.show()


if __name__ == "__main__":
    main()

