import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
class LogisticRegession:
    def sigmoid(self, x):
        y_prob = 1.0 / (1.0 + np.exp(-x))
        return y_prob

    def predict_prob(self, x):
        y_prob = self.sigmoid(np.dot(x, self.w) + self.b)
        return y_prob

    def predict(self, X):
        inst_num = X.shape[0]
        probs = self.predict_prob(X)
        labels = np.zeros(inst_num)
        for i in range(inst_num):
            if probs[i] >= 0.5:
                labels[i] = 1
        return probs, labels

    def loss_function(self, train_x, train_y):
        inst_num = train_x.shape[0]
        loss = 0.0
        for i in range(inst_num):
            z = np.dot(train_x[i, :], self.w) + self.b
            loss += -train_y[i] * z + np.log(1 + np.exp(z))  # see Eq.(2.10)
        loss = loss / inst_num
        return loss

    def calculate_grad(self, train_x, train_y):
        inst_num = train_x.shape[0]  # data size
        probs = self.sigmoid(train_x.dot(self.w) + self.b)
        grad_w = np.sum(np.dot(train_x.T, (probs - train_y))) / inst_num
        grad_b = np.sum(probs - train_y) / inst_num
        return grad_w, grad_b

    def gradient_descent(self, train_x, train_y, learn_rate, max_iter, epsilon):
        loss_list = []
        for i in range(max_iter):
            loss_old = self.loss_function(train_x, train_y)
            loss_list.append(loss_old)
            grad_w, grad_b = self.calculate_grad(train_x, train_y)
            self.w = self.w - learn_rate * grad_w
            self.b = self.b - learn_rate * grad_b
            loss_new = self.loss_function(train_x, train_y)
            if abs(loss_new - loss_old) <= epsilon:
                break
        return loss_list

    def fit(self, train_x, train_y, learn_rate, max_iter, epsilon):
        feat_num = train_x.shape[1]  # feature dimension
        self.w = np.zeros((feat_num, 1))  # initialize model parameters
        self.b = 0.0
        # learn model parameters using gradient descent algorithm
        loss_list = self.gradient_descent(train_x, train_y, learn_rate, max_iter
                                          , epsilon)
        self.training_visualization(loss_list)

    def training_visualization(self, loss_list):
        plt.plot(loss_list, color='red')
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.savefig("loss.png", bbox_inches= 'tight', dpi = 400)
        plt.show()
if __name__ == '__main__':
    data, label = make_blobs(n_samples=200, n_features=2, centers=2)
    train_x = np.array(data)
    label = np.array(label)
    train_label = label.reshape(-1, 1)
    LR = LogisticRegession()
    LR.fit(data, train_label, 0.01, 1000, 0.00001)
    df = pd.DataFrame()
    df['x1']=data[:, 0]
    df['x2']=data[:, 1]
    df['class']=label
    positive = df[df["class"] == 1]
    negative = df[df["class"] == 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(positive["x1"], positive["x2"], s=30, c="b", marker="o", label="class 1")
    ax.scatter(negative["x1"], negative["x2"], s=30, c="r", marker="x", label="class 0")
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    orig_data = df.values
    cols = orig_data.shape[1]
    data_mat = orig_data[:, 0:cols - 1]
    a = min(data_mat[:, 0])
    b = max(data_mat[:, 0])
    lin_x = np.linspace(a, b, 200)
    lin_y = (-float(LR.b) - LR.w[0, 0] * lin_x) / LR.w[1, 0]
    plt.plot(lin_x, lin_y, color="red")
    plt.savefig("result.png", bbox_inches= 'tight', dpi = 400)
    plt.show()
