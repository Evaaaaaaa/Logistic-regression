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
    # print clients.corr()
    # clients.head(15)

    clients.columns = ['memNum','productNum','routine','date','timeInt','quantity','memDay', 'sale']
    # clients['routine'] = clients['routine'].astype(int)

    feature_col = clients.loc[:,'quantity']
    target_col = clients.loc[:,'timeInt']

    # print feature_col.head()
    # print target_col.head()
    # sf = []
    # st = []
    # for i in range(len(target_col)/5):
    #     sf.append(feature_col[5*i]+feature_col[5*i+1]+feature_col[5*i+2]+feature_col[5*i+3]+feature_col[5*i+4])
    #     st.append(target_col[5*i]+ target_col[5*i+1]+target_col[5*i+2]+target_col[5*i+3]+target_col[5*i+4])
    #
    # print sf
    # print st

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


# sklearn linear regression
def log_reg(clients_data, clients_target):

    # change list into array then reshape it
    X = np.array(clients_data).reshape(-1,1)
    y = clients_target

    # split training and testing set
    X_test = X[-1]
    y_test = y[-1]
    X_train = X[:-1]
    y_train = y[:-1]

    print X_train.shape
    print X_test.shape

    logReg = LogisticRegression()
    model1 = logReg.fit(X_train, y_train)

    # print model
    # print logReg.intercept_
    # print logReg.coef_

    # pair the feature names with the coefficients
    # print zip(feature_cols, logReg.coef_)

    y_pred_train = logReg.predict(X_train)
    y_pred = logReg.predict(X)
    pred_lp = model1.predict(X_test)
    print pred_lp
    print y_test
    print pred_lp - y_test

    plt.figure()
    plt.plot(range(len(y_pred_train)), y_pred_train, 'b', label="predict")
    plt.plot(range(len(y_pred)), y, 'r', label="test")
    plt.legend(loc="upper right")
    plt.ylabel('time Interval')
    plt.xlabel('quantity')
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

