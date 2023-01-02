import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):

        self.num_layers = len(layer_sizes)  # layer number of NN
        self.layers = layer_sizes  # node numbers of each layer
        self.weights = [0.001*np.random.randn(y,x) for x,y in zip(layer_sizes[:-1],layer_sizes[1:])]
        self.biases = [0.01*np.random.randn(y,1) for y in layer_sizes[1:]]

    def sigmoid(self, z):
        act = 1.0 / (1.0 + np.exp(-z))
        return act

    def sigmoid_prime(self, z):
        act = self.sigmoid(z) * (1.0 - self.sigmoid(z))
        return act

    def tanh(self, z):
        act = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return act

    def tanh_prime(self, z):
        return 1 - self.relu(z)**2

    def relu(self, z):
        return np.maximum(0.0, z)

    def relu_prime(self, z):
        z[z<=0] = 0
        z[z>0] = 1

    def feed_forward(self, x):
        output = x.copy()
        for w, b in zip(self.weights, self.biases):
            output = self.tanh(np.dot(w, output) + b)
        return output

    def feed_backward(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = np.transpose(x)
        activations = [activation]
        layer_input = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            layer_input.append(z)  # input of each layer
            activation = self.tanh(z)
            activations.append(activation)  # output of each layer

        # loss function
        ground_truth = np.transpose(y)
        diff = activations[-1] - ground_truth
        # get input of last layer
        last_layer_input = layer_input[-1]
        delta = np.multiply(diff, self.tanh_prime(last_layer_input))
        # bias update of last layer
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        # weight update of last layer
        delta_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # update weights and bias from 2nd layer to last layer
        for i in range(2, self.num_layers):
            input_values = layer_input[-i]
            delta = np.multiply(np.dot(np.transpose(self.weights[-i + 1]), delta), self.tanh_prime(input_values))
            delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
            delta_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
        return delta_b, delta_w

    def fit(self, x, y, lr, mini_batch_size, epochs=1000):
        n = len(x)  # training size
        for i in range(epochs):
            # if i%100 == 0:
            print("=" * 10 + f" Epoch {i + 9999} " + "=" * 10)
            random_list = np.random.randint(0, n - mini_batch_size, int(n / mini_batch_size))
            batch_x = [x[k:k + mini_batch_size] for k in random_list]
            batch_y = [y[k:k + mini_batch_size] for k in random_list]
            for j in range(len(batch_x)):
                delta_b, delta_w = self.feed_backward(batch_x[j], batch_y[j])
                self.weights = [w - (lr / mini_batch_size) * dw for w, dw in
                                zip(self.weights, delta_w)]
                self.biases = [b - (lr / mini_batch_size) * db for b, db in
                               zip(self.biases, delta_b)]

            labels = self.predict(x)
            acc = 0.0
            for k in range(len(labels)):
                if y[k, labels[k]] == 1.0:
                    acc += 1.0
            acc = acc / len(labels)
            # if i%100 == 0:
            print("epoch train %d accuracy %.3f" % (i + 9999, acc))

    def predict(self, x):
        results = self.feed_forward(x.T)
        labels = [np.argmax(results[:, y]) for y in range(results.shape[1])]
        return labels
