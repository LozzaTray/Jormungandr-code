# adapted from http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import math


class SoftmaxNeuralNet:


    def __init__(self, layers_size, sigma=1, a=250, b=1000, gamma=0.8):
        """Initialise Neural Network"""
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.sigma = 1
        self.a = a
        self.b = b
        self.gamma = gamma


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))


    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # keep numerically stable / avoid overflow
        expZ = np.nan_to_num(expZ)
        return expZ / expZ.sum(axis=0, keepdims=True)


    def _initialize_parameters(self):

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
        store["b" + str(self.L)] = self.parameters["b" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store


    def _sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)


    def _backward_cross_entropy_loss(self, X, Y, store):

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


    def _log_prior_derivative(self, store):
        derivatives = {}

        for l in range(1, self.L + 1):
            derivatives["dW" + str(l)] = - store["W" + str(l)] / self.sigma
            derivatives["db" + str(l)] = - store["b" + str(l)] / self.sigma

        return derivatives

    
    def _backward_log_posterior(self, X, Y, store):
        log_lik_derivatives = self._backward_cross_entropy_loss(X, Y, store)
        log_prior_derivatives = self._log_prior_derivative(store)

        log_post_derivatives = {}
        for key in log_lik_derivatives:
            log_post_derivatives[key] = self.n * log_lik_derivatives[key] + log_prior_derivatives[key]

        return log_post_derivatives


    def sgld_initialise(self, input_dimension):
        """Initilaise SGLD - Stochastic Gradient Langevin Diffusion for MCMC sampling form posterior"""
        self.layers_size.insert(0, input_dimension)
        self._initialize_parameters()
        self.weight_history = {}
        self.bias_history = {}
        self.t = 0

        for l in range(1, len(self.layers_size)):
            self.weight_history["W" + str(l)] = []
            self.bias_history["b" + str(l)] = []

    
    def cross_entropy_loss(self, X, Y):
        A, _store = self._forward(X)
        cost = -np.mean(Y * np.log(A.T + 1e-8))
        return cost

    
    def sgld_iterate(self, X, Y, step_scaling=1):
        """Perform one iteration of sgld, returns previous cost"""
        self.n = X.shape[0]
        step_size = step_scaling * self.anneal_step_size(self.t)
        self.t += 1

        A, store = self._forward(X)
        cost = -np.mean(Y * np.log(A.T + 1e-8))
        derivatives = self._backward_log_posterior(X, Y, store)


        for l in range(1, self.L + 1):
            weight_shape = self.parameters["W" + str(l)].shape
            weight_noise = np.sqrt(step_size) * np.random.randn(*weight_shape) # * unpacks tuple into arg list
            new_weight = self.parameters["W" + str(l)] - step_size * derivatives["dW" + str(l)] / 2 + weight_noise

            self.parameters["W" + str(l)] = new_weight
            self.weight_history["W" + str(l)].append(new_weight)

            bias_shape = self.parameters["b" + str(l)].shape
            bias_noise = np.sqrt(step_size) * np.random.randn(*bias_shape)
            new_bias = self.parameters["b" + str(l)] - step_size * derivatives["db" + str(l)] / 2 + bias_noise

            self.parameters["b" + str(l)] = new_bias
            self.bias_history["b" + str(l)].append(new_bias)

        return cost


    def anneal_step_size(self, t):
        return self.a * math.pow(self.b + t, -1 * self.gamma)


    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500, seed=None, verbose=False):
        """
        Train params of Neural Net:
            X : (n, d) array - where n is num training points and d is dimension (num features)
            Y : (n, k) array - one-hot encoding of class labels
            learning_rate - speed of gradient ascent
            n_iterations - num iterations
        """
        if len(self.costs) > 0:
            print("Classifier already trained must create new instance >> ABORTING")
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
            derivatives = self._backward_cross_entropy_loss(X, Y, store)

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

    
    def plot_final_weights(self, feature_names):
        W = self.parameters["W" + str(self.L)]

        B = W.shape[0]
        D = W.shape[1]        

        assert len(feature_names) == D

        width = 0.8 / D

        for d in range(0, D):
            y = []
            for b in range(0, B):
                y.append(W[b, d])

            x = np.array(range(0, B)) + (width * d)
            plt.bar(x, y, width=width, label=feature_names[d])
        
        plt.title("ML softmax fit")
        plt.xlabel("Class label")
        plt.ylabel("Weight")
        plt.legend()
        plt.show()


    def plot_sampled_weights(self, feature_names):
        W_history = self.weight_history["W" + str(self.L)]

        B = W_history[0].shape[0]
        D = W_history[0].shape[1]
        n = len(W_history)       

        assert len(feature_names) == D

        width = 0.4 / D

        for d in range(0, D):
            mean = np.zeros(B)
            square = np.zeros(B)
            
            for W in W_history:
                mean[:] += W[:, d]
                square[:] += W[:, d] ** 2

            mean = mean / n
            square = square / n

            std_dev = np.sqrt(square - mean ** 2)

            height = 2 * std_dev
            bottom = mean - std_dev

            x = np.array(range(0, B)) + (width * d)
            plt.bar(x=x, height=height, bottom=bottom, width=width, label=feature_names[d])
        
        plt.title("Sampled softmax weightings")
        plt.xlabel("Class label")
        plt.ylabel("Weight mean $\\pm \\sigma$")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_sample_histogram(self):
        W_history = self.weight_history["W" + str(self.L)]
        n = len(W_history)
        values = [w[0, 0] for w in W_history]     
        plt.hist(values)
        
        plt.title("Weight samples (n={})".format(n))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

    
    def plot_sample_history(self):
        W_history = self.weight_history["W" + str(self.L)]
        
        B = W_history[0].shape[0]
        D = W_history[0].shape[1]
        n = len(W_history)       

        for b in range(0, B):
            mean = [np.mean(w[b, :]) for w in W_history]
            plt.plot(mean, label="block-{}".format(b))
        
        plt.title("Sampled softmax weightings")
        plt.xlabel("Sample index")
        plt.ylabel("Weight")
        plt.grid()
        plt.legend()
        plt.show()


def from_values_to_one_hot(y):
    y = np.array(y)
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_hot = enc.fit_transform(y.reshape(len(y), -1))
    return y_hot
