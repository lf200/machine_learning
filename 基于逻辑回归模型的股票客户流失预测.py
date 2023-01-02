import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Logistic_Regression import LogisticRegession as LR
from sklearn.linear_model import LogisticRegression as LR_SKL
f = open('C:\pythonprogram\machine_learning\dataset\Stock_Client_loss.csv')
data = pd.read_csv(f)
data_x = data[["x1","x2","x3","x4","x5"]]
data_y = np.array(data["loss"])
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)
X_train, X_test, Y_train, Y_test = train_test_split(data_x,data_y,test_size=0.3,shuffle=True)
learnrate = 0.01
maxiter = 1000
eps = 1e-5
def cal_acc(y_test,y_pred):
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            acc += 1.0
    return acc/len(y_test)
if __name__ == '__main__':
    LR_model = LR()
    LR_model.fit(train_x=X_train,train_y=Y_train.reshape(-1,1),learn_rate=learnrate,max_iter=maxiter, epsilon=eps)
    _,Y_test_pred_LR = LR_model.predict(X_test)
    acc = cal_acc(Y_test,Y_test_pred_LR)
    print("LR ACC:%.3f"%(acc))
    LR_SKL_model = LR_SKL()
    LR_SKL_model.fit(X_train,Y_train)
    Y_test_pred_LR_SKL = LR_SKL_model.predict(X_test)
    acc = cal_acc(Y_test,Y_test_pred_LR_SKL)
    print("LR_SKL ACC:%.3f"%(acc))