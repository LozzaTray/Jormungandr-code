# adapted from http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
import math
from inference.store import Store


class SoftmaxNeuralNet:

    def __init__(self, layers_size, sigma=1, a=250, b=1000, gamma=0.8):
        """Initialise Neural Network"""
        self.layers_size = layers_size
        self.parameters = Store()
        self.L = len(self.layers_size) - 1  # number of activation layers
        self.n = 0
        self.costs = []
        self.sigma = 1
        self.a = a
        self.b = b
        self.gamma = gamma
        self._initialize_parameters()

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _softmax(self, Z):
        # keep numerically stable / avoid overflow
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        expZ = np.nan_to_num(expZ)
        return expZ / expZ.sum(axis=0, keepdims=True)

    def _initialize_parameters(self):

        for l in range(1, len(self.layers_size)):
            W = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(self.layers_size[l - 1])
            b = np.zeros((self.layers_size[l], 1))

            self.parameters.set_W(W, l)
            self.parameters.set_b(b, l)

    def _forward(self, X):
        store = Store()

        A = X.T
        store.set_A(A, 0)
        # hidden sigmoid layers
        for l in range(1, self.L + 1):
            Z = self.parameters.get_W(l).dot(A) + self.parameters.get_b(l)

            if l < self.L:
                A = self._sigmoid(Z)
            else:
                A = self._softmax(Z)  # final softmax layer

            store.set_A(A, l)
            store.set_W(self.parameters.get_W(l), l)
            store.set_b(self.parameters.get_b(l), l)
            store.set_Z(Z, l)

        return A, store

    def _sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def _backward_cross_entropy_deriv(self, X, Y, store):

        derivatives = {}

        A = store.get_A(self.L)
        dZ = A - Y.T

        dW = dZ.dot(store.get_A(self.L - 1).T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store.get_W(self.L).T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self._sigmoid_derivative(store.get_Z(l))
            dW = 1. / self.n * dZ.dot(store.get_A(l-1).T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store.get_W(l).T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def _log_prior_derivative(self, store):
        derivatives = {}

        for l in range(1, self.L + 1):
            derivatives["dW" + str(l)] = - store.get_W(l) / self.sigma
            derivatives["db" + str(l)] = - store.get_b(l) / self.sigma

        return derivatives

    def _backward_log_posterior_deriv(self, X, Y, store):
        log_lik_derivatives = self._backward_cross_entropy_deriv(X, Y, store)
        log_prior_derivatives = self._log_prior_derivative(store)

        log_post_derivatives = {}
        for key in log_lik_derivatives:
            log_post_derivatives[key] = self.n * \
                log_lik_derivatives[key] - log_prior_derivatives[key]

        return log_post_derivatives

    def _log_target(self, store, A, Y):
        log_posterior = np.mean(Y * np.log(A.T + 1e-8))  # -ve of cross-entropy
        log_prior = 0
        for l in range(1, self.L+1):
            weight_term = norm.logpdf(store.get_W(l), scale=self.sigma).sum()
            bias_term = norm.logpdf(store.get_b(l), scale=self.sigma).sum()
            log_prior = log_prior + weight_term + bias_term

        return - (log_posterior + log_prior)

    def sgld_initialise(self):
        """Initilaise SGLD - Stochastic Gradient Langevin Diffusion for MCMC sampling form posterior"""
        self.weight_history = {}
        self.bias_history = {}
        self.t = 0

        for l in range(1, len(self.layers_size)):
            self.weight_history["W" + str(l)] = []
            self.bias_history["b" + str(l)] = []

    def transition_log_pdf_diff(store_0, store_1, step_size):
        distances = []
        for l in range(1, L+1):
            raise NotImplementedError

    def mala_perform(self, X, Y, num_iter=1000, step_scaling=1, verbose=False):
        """Performs Metropolis-Adjusted Langevin Algorithm

            Parameters:
                X (int[][]): n x D matrix of feature flags
                Y (int[][]): n x B matrix os posterior probs
                num_iter (int): number of iterations to run
            Returns:
                None
        """
        self.n = X.shape[0]

        A, store = self._forward(X)
        cost = -np.mean(Y * np.log(A.T + 1e-8))
        derivatives = self._backward_log_posterior_deriv(X, Y, store)

        for t in range(0, num_iter):
            step_size = step_scaling * self.anneal_step_size(t)

            A, store = self._forward(X)
            U = self._log_target(store, A, Y)
            derivatives = self._backward_log_posterior_deriv(X, Y, store)

            for l in range(1, self.L + 1):
                # simulate
                pass

            new_A, new_store = self._forward(X)
            new_U = self._log_target(new_store, new_A, Y)

            accepted = True

            if accepted:
                for l in range(1, self.L+1):
                    pass

            if t % 10 == 0 and verbose:
                print(cost)

    def sgld_iterate(self, X, Y, step_scaling=1):
        """Perform one iteration of sgld, returns previous cost"""
        self.n = X.shape[0]
        step_size = step_scaling * self.anneal_step_size(self.t)
        self.t += 1

        A, store = self._forward(X)
        cost = -np.mean(Y * np.log(A.T + 1e-8))
        derivatives = self._backward_log_posterior_deriv(X, Y, store)

        for l in range(1, self.L + 1):
            weight_shape = self.parameters.get_W(l).shape
            # * unpacks tuple into arg list
            weight_noise = np.sqrt(step_size) * np.random.randn(*weight_shape)
            new_weight = self.parameters.get_W(l) - step_size * \
                derivatives["dW" + str(l)] / 2 + weight_noise

            self.parameters.set_W(new_weight, l)
            self.weight_history["W" + str(l)].append(new_weight)

            bias_shape = self.parameters.get_b(l).shape
            bias_noise = np.sqrt(step_size) * np.random.randn(*bias_shape)
            new_bias = self.parameters.get_b(l) - step_size * \
                derivatives["db" + str(l)] / 2 + bias_noise

            self.parameters.set_b(new_bias, l)
            self.bias_history["b" + str(l)].append(new_bias)

        return cost

    def sgld_sample_thinning(self, burn_in_pc=10, thinning_pc=20):
        """discard samples pre burn-in and only select keep thinning_pc of rest"""
        stop = len(self.weight_history["W" + str(self.L)])
        start = int(stop * burn_in_pc / 100)
        step = int(100 / thinning_pc)

        for l in range(1, self.L + 1):
            self.weight_history["W" + str(self.L)] = self.weight_history["W" +
                                                                         str(self.L)][start:stop:step]
            self.bias_history["b" + str(self.L)] = self.bias_history["b" +
                                                                     str(self.L)][start:stop:step]

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

        for loop in range(n_iterations):
            A, store = self._forward(X)
            cost = -np.mean(Y * np.log(A.T + 1e-8))
            derivatives = self._backward_cross_entropy_deriv(X, Y, store)

            for l in range(1, self.L + 1):
                W = self.parameters.get_W(l) - learning_rate * derivatives["dW" + str(l)]
                b = self.parameters.get_b(l) - learning_rate * derivatives["db" + str(l)]

                self.parameters.set_W(W, l)
                self.parameters.set_b(b, l)

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

    def compute_mean_variances(self):
        D = self.layers_size[0]
        B = self.layers_size[1]

        param_means = np.zeros(shape=(B, D+1))
        param_std_devs = np.zeros(shape=(B, D+1))

        W_history = self.weight_history["W1"]
        b_history = self.bias_history["b1"]

        B = W_history[0].shape[0]
        D = W_history[0].shape[1]
        n = len(W_history)

        assert D == W_history[0].shape[1]
        assert B == W_history[0].shape[0]

        for d in range(0, D + 1):
            mean = np.zeros(B)
            square = np.zeros(B)

            if d == D:
                for b in b_history:
                    mean[:] += b[:, 0]
                    square[:] += b[:, 0] ** 2

            else:
                for W in W_history:
                    mean[:] += W[:, d]
                    square[:] += W[:, d] ** 2

            mean = mean / n
            square = square / n

            std_dev = np.sqrt(square - mean ** 2)

            param_means[:, d] = mean
            param_std_devs[:, d] = std_dev

        self.param_means = param_means
        self.param_std_devs = param_std_devs

    def feature_overlaps_zero(self, feature_index, std_dev_multiplier):
        means = self.param_means[:, feature_index]
        std_devs = self.param_std_devs[:, feature_index]

        lower = means - (std_dev_multiplier * std_devs)
        upper = means + (std_dev_multiplier * std_devs)

        for i in range(0, len(lower)):
            if lower[i] < 0 and upper[i] > 0:
                pass
            else:
                return False

        return True

    def feature_overlaps_zero_for_class(self, feature_index, class_index, std_dev_multiplier):
        mean = self.param_means[class_index, feature_index]
        std_dev = self.param_std_devs[class_index, feature_index]

        lower = mean - (std_dev_multiplier * std_dev)
        upper = mean + (std_dev_multiplier * std_dev)
        if lower[i] < 0 and upper[i] > 0:
            return True
        else:
            return False

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()

    def plot_final_weights(self, feature_names):
        W = self.parameters.get_W(self.L)

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

    def plot_sampled_weights(self, feature_names, std_dev_multiplier=1, B_range=None):
        """Plot the sampled weights.

            Parameters:
                feature_names (str[]): names of each feature
                std_dev_multiplier (float): how many std devs to consider when disregarding
                B_range ((int, int)): tuple of ints specifying lower and upper B cutoff

            Returns:
                None. Just plot the samples
        """

        D = self.layers_size[0]
        if B_range is None:
            B_range = (0, self.layers_size[1])

        B = B_range[1] - B_range[0]

        assert len(feature_names) == D

        feature_names.append("bias")

        self.compute_mean_variances()

        width = 0.4 / (D + 1)
        midpoint = D / 2.0

        discarded_features = []
        for d in range(0, D + 1):
            if self.feature_overlaps_zero(d, std_dev_multiplier):
                discarded_features.append(d)
                print("Discarding feature {}: {}".format(d, feature_names[d]))

        b_counter = np.zeros(B)
        for d in range(0, D + 1):
            kept_x = []
            kept_height = []
            kept_bottom = []

            if d in discarded_features:
                pass
            else:
                mean = self.param_means[:, d][B_range[0]: B_range[1]]
                std_dev = self.param_std_devs[:, d][B_range[0]: B_range[1]]

                height = 2 * std_dev * std_dev_multiplier
                bottom = mean - std_dev

                for b in range(B_range[0], B_range[1]):
                    if bottom[b] < 0 and bottom[b] + height[b] > 0:
                        height[b] = 0  # unclutter classifier
                    else:
                        kept_x.append(b + b_counter[b] * width)
                        kept_height.append(height[b])
                        kept_bottom.append(bottom[b])
                        b_counter[b] += 1

                x = np.arange(0, B, 1) + (width * (d - midpoint))
                plt.bar(x=x, height=height, bottom=bottom,
                        width=width, label=feature_names[d])
                #plt.bar(x=kept_x, height=kept_height, bottom=kept_bottom, width=width, label=feature_names[d])

        block_centres = np.arange(0, B, 1)
        block_names = [str(num) for num in range(0, B)]

        plt.title("Sampled softmax weightings")
        plt.xlabel("Class label")
        plt.ylabel("Weight mean $\\pm {}\\sigma$".format(std_dev_multiplier))
        plt.grid()
        plt.legend()
        plt.xticks(ticks=block_centres, labels=block_names)
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
