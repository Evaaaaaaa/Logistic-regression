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
    address = '/Users/Evangeline0519/PycharmProjects/Logistic_Regression/one_mem_buy_data.csv'
    clients = pd.read_csv(address)
    # clients.head(15)

    clients.columns = ['memNum','productNum','routine','date','timeInt','quantity','memDay', 'sale']
    # clients['routine'] = clients['routine'].astype(int)

    feature_cols = clients.loc[:,('routine','quantity','memDay','sale')]
    clients_data = feature_cols.values
    clients_target = clients.loc[:,'timeInt'].values
    # print feature_cols.head()
    # print feature_cols.shape
    # print clients.loc[:,'timeInt'].head()

    pair_plot(clients)
    # log_reg(clients_data, clients_target, feature_cols)


# seaborn pair plots for each feature
def pair_plot(clients):
    sns.pairplot(clients, x_vars=['routine','quantity','memDay','sale'], y_vars='timeInt', size=7, aspect=0.8,kind='reg')
    clients_data_names = ['timeInt','routine','quantity','memDay','sale']
    plt.show()
    return


# sklearn linear regression
def log_reg(clients_data, clients_target,feature_cols):

    logReg = LogisticRegression()
    X = clients_data
    y = clients_target

    model = logReg.fit(X, y)
    # print model
    # print logReg.intercept_
    print logReg.coef_

    # pair the feature names with the coefficients
    print zip(feature_cols, logReg.coef_)

    y_pred = logReg.predict(X)

    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y, 'r', label="test")
    plt.legend(loc="upper right")
    plt.ylabel('time Interval')
    plt.show()

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




