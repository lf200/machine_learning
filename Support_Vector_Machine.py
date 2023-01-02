import random
import numpy as np
class SupportVectorMachine:
    def __init__(self, x_train, y_train, C, toler, max_iter):
        """
        param: x_train:matrix of feature vectors
        param: y_train:labels of training instaces
        param: C: C>0
        param: toler: alowed error
        param: max_iter: max iterations
        """

        self.x_train = np.mat(x_train)
        self.y_train = np.mat(y_train)
        self.C = C
        self.toler = toler
        self.max_iter = max_iter
        m, n = np.shape(self.x_train)
        self.alphas = np.zeros((m, 1))
        self.b = 0

    def rand_select_j(self, i, m):
        j = i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H

        if L > aj:
            aj = L
        return aj

    def smo(self):
        iters = 0

        m, n = np.shape(self.x_train)
        while iters < self.max_iter:
            alpha_pairs_changed = 0
            for i in range(m):
                W_i = np.dot(np.multiply(self.alphas, self.y_train).T, self.x_train)
                f_x_i = float(np.dot(W_i, self.x_train[i, :].T)) + self.b
                E_i = f_x_i - float(self.y_train[i])
                if ((self.y_train[i] * E_i < -self.toler) and (self.alphas[i] < self.C)) or \
                        ((self.y_train[i]*E_i > self.toler) and (self.alphas[i] > 0)):
                    j = self.rand_select_j(i, m)
                    W_j = np.dot(np.multiply(self.alphas, self.y_train).T,
                                 self.x_train)
                    f_x_j = float(np.dot(W_j, self.x_train[j, :].T)) + self.b
                    E_j = f_x_j - float(self.y_train[j])
                    alpha_iold = self.alphas[i].copy()
                    alpha_jold = self.alphas[j].copy()
                    if (self.y_train[i] != self.y_train[j]):
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    if H == L: continue
                    eta = 2.0 * self.x_train[i, :] * self.x_train[j, :].T - self.x_train[i, :] * self.x_train[i, :].T - \
                    self.x_train[j, :] * self.x_train[j, :].T
                    if eta >= 0: continue
                    self.alphas[j] = (self.alphas[j] - self.y_train[j] * (E_i -
                                                                          E_j)) / eta
                    self.alphas[j] = self.clip_alpha(self.alphas[j], H, L)
                    if (abs(self.alphas[j] - alpha_jold) < 0.00001):
                        continue
                    self.alphas[i] = self.alphas[i] + self.y_train[j] * self.y_train[i] * (alpha_jold - self.alphas[j])
                    b1 = self.b - E_i + self.y_train[i] * (alpha_iold - self.alphas[i]) * np.dot(self.x_train[i, :], self.x_train[i, :].T) + \
                         self.y_train[j] * (alpha_jold - self.alphas[j]) * np.dot(self.x_train[i, :], self.x_train[j, :].T)
                    b2 = self.b - E_j + self.y_train[i] * (alpha_iold - self.alphas[i]) * np.dot(self.x_train[i, :], self.x_train[j, :].T) + \
                         self.y_train[j] * (alpha_jold - self.alphas[j]) * np.dot(self.x_train[j, :], self.x_train[j, :].T)
                    if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                        self.b = b1
                    elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    alpha_pairs_changed += 1
                if (alpha_pairs_changed == 0):
                    iters += 1
                else:
                    iters = 0





