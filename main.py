import pandas as pd
import numpy as np
from Linear_Regression import LinearRegression as LR
from sklearn.linear_model import LinearRegression as LR_SKL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
f = open('D:\sjgc\Client_Value.csv')
data = pd.read_csv(f)
scaler = StandardScaler()
data = scaler.fit_transform(data)
X_train, X_test, Y_train, Y_test = train_test_split(data[:, 1:],
                                                    data[:, 0],
                                                    test_size=0.3,
                                                    shuffle=True)
X_train = X_train.reshape(-1,5)
Y_train = Y_train.reshape(-1,1)
X_test = X_test.reshape(-1,5)
Y_test = Y_test.reshape(-1,1)
learnrate = 0.01
maxiter = 1000
eps = 1e-5
def cal_rmse(y_test,y_pred):
    loss = np.sum((y_pred - y_test)**2)/(len(y_test))
    return np.sqrt(loss)
LR_model = LR()
LR_model.fit(train_x=X_train,train_y=Y_train,
            learn_rate=learnrate,max_iter=maxiter, epsilon=eps)
Y_test_pred_LR = LR_model.predict(X_test)
rmse = cal_rmse(Y_test,Y_test_pred_LR)
print("LR_model RMSE:%.3f"%(rmse))
# LR_batch_model = LR()
# LR_batch_model.fit_batch(train_x=X_train,train_y=Y_train,learn_rate=learnrate,max_iter=maxiter,epsilon=eps)
# Y_test_pred_LR_batch = LR_batch_model.predict(X_test)
# rmse = cal_rmse(Y_test,Y_test_pred_LR_batch)
# print("LR_model_batch RMSE :%.3f"%(rmse))
LR_SKL_model = LR_SKL()
LR_SKL_model.fit(X_train,Y_train)
Y_test_pred_LR_SKL = LR_SKL_model.predict(X_test)
rmse = cal_rmse(Y_test,Y_test_pred_LR_SKL)
print("LR_SKL_model RMSE:%.3f"%(rmse))
