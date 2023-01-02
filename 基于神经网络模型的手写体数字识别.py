import struct
import numpy as np
import os
from Neural_Network import NeuralNetwork
#fucntion to load MNIST data
def load_mnist_data(directory, kind='train'):
    label_path = os.path.join(os.getcwd(), directory, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(os.getcwd(), directory, '%s-images.idx3-ubyte' % kind)
    with open(label_path, 'rb') as lbpath:  # open label file
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:  # open image file
        struct.unpack('>IIII', imgpath.read(16))
        # transform image into 784-dimensional feature vector
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
import matplotlib.pyplot as plt
def show_image(image):
    plt.figure()
    img = image.reshape(28,28)
    plt.imshow(img, 'gray')
    plt.show()
from sklearn.preprocessing import StandardScaler
path = 'C:\\pythonprogram\machine_learning\dataset\\MNIST'
train_images, train_labels = load_mnist_data(path, kind='train')
train_y = np.zeros((len(train_labels), 10))
for i in range(len(train_labels)):
    train_y[i, train_labels[i]] = 1
scaler = StandardScaler()
train_x = scaler.fit_transform(train_images)
test_images, test_labels = load_mnist_data(path, kind='t10k')
test_y = np.zeros((len(test_labels), 10))
for i in range(len(test_labels)):
    test_y[i, test_labels[i]] = 1
test_x = scaler.fit_transform(test_images)
layer_sizes = [784,100,10]
NN = NeuralNetwork(layer_sizes)
NN.fit(train_x, train_y, lr=0.01, mini_batch_size=100,epochs=3000)
test_pred_labels = NN.predict(test_x)   
acc = 0.0
for k in range(len(test_pred_labels)):
    if test_y[k,test_pred_labels[k]]==1.0:
        acc += 1.0
acc=acc/len(test_pred_labels)
print("test accuracy:%.3f"%(acc))
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd',batch_size=100,learning_rate='constant',learning_rate_init=0.01,max_iter=3000)
model.fit(train_x, train_y)
labels = model.predict(test_x)
acc = 0.0
for k in range(len(labels)):
    index = 0
    for j in range(10):
        if labels[k,j]==1:
            index = j
            break
    if test_y[k,index]==1.0:
        acc += 1.0
acc=acc/len(labels)
print("test accuracy:%.3f"%(acc))

