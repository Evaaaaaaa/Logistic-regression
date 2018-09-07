import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

import sklearn
from sklearn.linear_model import LogisticRegression


def main():
    rcParams['figure.figsize'] = 5, 4
    sns.set_style('whitegrid')
    address = '/Users/Evangeline0519/Downloads/one_mem_buy_data.csv'
    clients = pd.read_csv(address)
    # clients.head(15)

    clients.columns = ['memNum','productNum','routine','date','timeInt','quantity','memDay', 'sale']
    # clients['routine'] = clients['routine'].astype(int)

    clients_data = clients.loc[:,'quantity'].values
    clients_target = clients.loc[:,'timeInt'].values

    # print clients.loc[:,'timeInt'].head()
    # print clients.loc[:,'quantity'].head()

    # pair_plot(clients)
    log_reg(clients_data, clients_target)



# seaborn pair plots for each feature
def pair_plot(clients):
    sns.pairplot(clients, x_vars=['routine','quantity','memDay','sale'], y_vars='timeInt', size=7, aspect=0.8,kind='reg')
    clients_data_names = ['timeInt','routine','quantity','memDay','sale']
    plt.show()
    return


# sklearn linear regression with training
def log_reg(clients_data, clients_target):

    X = clients_data.reshape(-1, 1)
    y = clients_target.ravel()

    # split training and testing set
    X_test = X[-1]
    y_test = y[-1]
    X_train = X[:-1]
    y_train = y[:-1]

    print X_train.shape
    print X_test.shape

    # fit in linear regression model
    logReg = LogisticRegression()

    model1 = logReg.fit(X_train, y_train)
    model2 = logReg.fit(X, y)
    # print model
    # print logReg.intercept_
    # print logReg.coef_

    # pair the feature names with the coefficients
    # print zip(feature_cols, logReg.coef_)

    y_pred_train = logReg.predict(X_train)
    y_pred = logReg.predict(X)
    pred_lp =  model1.predict(X_test)
    print pred_lp
    print y_test
    print pred_lp - y_test


    plt.figure()
    # plt.scatter(X, y, 60, color='blue', marker='o', linewidth=0.1, alpha=0.8)
    # plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred_train)), y_pred_train, 'b', label="predict with training")
    plt.plot(range(len(y_pred)), y, 'r', label="test")
    plt.legend(loc="upper right")
    plt.ylabel('time Interval')
    plt.xlabel('quantity')
    plt.show()

    return


def evaluate(logReg, X_train, X_test, y_train, y_test):
    print "predicted y: "
    print list(logReg.predict(X_test))
    print "real y: "
    print list(y_test)
    print "variance: "
    print ((y_test - logReg.predict(X_test)) ** 2).sum()

    return


if __name__ == '__main__':
    main()


# #create a python list of feature names
# X = clients_data
# # print the first 5 rows
# print X.head()
# # check the type and shape of X
# print type(X)
# print X.shape
#
# # select a Series from the DataFrame
# y = clients_target
# # print the first 5 values
# print y.head()
#

