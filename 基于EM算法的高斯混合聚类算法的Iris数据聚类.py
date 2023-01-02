from sklearn import datasets
import numpy as np
from 基于EM算法的高斯混合聚类算法的实现 import GaussianMixtureClusterEM

iris = datasets.load_iris()
iris_x = iris.data
iris_label = iris.target
# just using the third and fourth feature
X = iris_x[:, [2, 3]]
# samples of calss0
X_0 = []
for i in range(len(iris_label)):
    if iris_label[i] == 0:
        X_0.append(X[i])
X_0 = np.array(X_0)
# samples of class1
X_1 = []
for i in range(len(iris_label)):
    if iris_label[i] == 1:
        X_1.append(X[i])
X_1 = np.array(X_1)
# samples of class2
X_2 = []
for i in range(len(iris_label)):
    if iris_label[i] == 2:
        X_2.append(X[i])
X_2 = np.array(X_2)
# show original data
import matplotlib.pyplot as plt

plt.scatter(X_0[:, 0], X_0[:, 1], label="class0", marker="o")
plt.scatter(X_1[:, 0], X_1[:, 1], label="class1", marker="x")
plt.scatter(X_2[:, 0], X_2[:, 1], label="class2", marker="+")
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("original sample distribution")
plt.legend()
plt.show()

model = GaussianMixtureClusterEM()
clusters, likelihoods, scores, sample_likelihoods, history = model.fit(X, n_clusters=3, n_epochs=500)
labels = []
for i in range(scores.shape[0]):
    max_value = scores[i, 0]
    max_index = 0
    for j in range(scores.shape[1]):
        if scores[i, j] > max_value:
            max_value = scores[i, j]
            max_index = j
    labels.append(max_index)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("GaussianMixtureClusterEM Clustering")
plt.legend()
plt.show()

from sklearn.mixture import GaussianMixture as GMM

gmm = GMM(n_components=3).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("sklearn GMM Clustering")
plt.legend()
plt.show()