# adapted from http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder


class SoftmaxNeuralNet:


    def __init__(self, layers_size):
        """Initialise Neural Network"""
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))


    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)


    def _initialize_parameters(self):
        np.random.seed(1)

        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))


    def _forward(self, X):
        store = {}

        A = X.T
        ## hidden sigmoid layers
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + \
                self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        ## final softmax layer
        Z = self.parameters["W" + str(self.L)].dot(A) + \
            self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store


    def _sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)


    def _backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self._sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives


    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500, seed=None):
        """
        Train params of Neural Net:
            X : (n, d) array - where n is num training points and d is dimension (num features)
            Y : (n, k) array - one-hot encoding of class labels
            learning_rate - speed of gradient ascent
            n_iterations - num iterations
        """
        if len(self.costs) > 0:
            print("Classifier already traiend must create new instance >> ABORTING")
            return

        if seed is not None:
            np.random.seed(seed)

        self.n = X.shape[0]

        if Y.shape[0] != self.layers_size[-1]:
            Y = from_values_to_one_hot(Y)

        # add input layer
        self.layers_size.insert(0, X.shape[1])
        self._initialize_parameters()
        for loop in range(n_iterations):
            A, store = self._forward(X)
            cost = -np.mean(Y * np.log(A.T + 1e-8))
            derivatives = self._backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]

            if loop % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

            if loop % 10 == 0:
                self.costs.append(cost)


    def predict(self, X, Y):
        A, _cache = self._forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100


    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def from_values_to_one_hot(y):
    y = np.array(y)
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_hot = enc.fit_transform(y.reshape(len(y), -1))
    return y_hot
