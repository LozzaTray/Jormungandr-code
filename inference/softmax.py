# adapted from http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/

import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
import math
from inference.store import Store, compute_log_acceptance_prob
from tqdm import tqdm
from utils.colors import plt_color


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
        self.store_history = []

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _softmax(self, Z):
        # keep numerically stable / avoid overflow
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        expZ = np.nan_to_num(expZ)
        return expZ / expZ.sum(axis=0, keepdims=True)

    def _initialize_parameters(self):

        for l in range(1, len(self.layers_size)):
            W = np.random.randn(
                self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(self.layers_size[l - 1])
            b = np.zeros((self.layers_size[l], 1))

            self.parameters.set_W(W, l)
            self.parameters.set_b(b, l)

    def _forward(self, X, store):
        A = X.T
        store.set_A(A, 0)
        # hidden sigmoid layers
        for l in range(1, self.L + 1):
            Z = store.get_W(l).dot(A) + store.get_b(l)

            if l < self.L:
                A = self._sigmoid(Z)
            else:
                A = self._softmax(Z)  # final softmax layer

            store.set_A(A, l)
            store.set_Z(Z, l)

        return A

    def _sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def _backward_cross_entropy_deriv(self, X, Y, store):
        """performs backwards algorithm on store object in place

            Parameters:
                X: input matrix for training
                Y: training target distribution
                store: store object with forward results

            Returns: 
                None - derivs added to store object
        """

        A = store.get_A(self.L)
        dZ = A - Y.T

        dW = dZ.dot(store.get_A(self.L - 1).T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store.get_W(self.L).T.dot(dZ)

        store.set_dW(dW, self.L)
        store.set_db(db, self.L)

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self._sigmoid_derivative(store.get_Z(l))
            dW = dZ.dot(store.get_A(l-1).T)
            db = np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store.get_W(l).T.dot(dZ)

            store.set_dW(dW, l)
            store.set_db(db, l)

    def _compute_grad_U(self, X, Y, store):
        """Computes derivatives of U w.r.t params W and b

            Parameters:
                X: input matrix for training
                Y: training target distribution
                store: store object with forward results

            Returns: 
                None - derivs added to store object
        """

        # first compute grad log-likelihood term
        self._backward_cross_entropy_deriv(X, Y, store)

        for l in range(1, self.L+1):
            dW_log_prior = - store.get_W(l) / (self.sigma ** 2)
            db_log_prior = - store.get_b(l) / (self.sigma ** 2)

            dW = store.get_dW(l) - dW_log_prior
            db = store.get_db(l) - db_log_prior

            store.set_dW(dW, l)
            store.set_db(db, l)

    def _compute_log_target(self, store, A, Y):
        """Computes -ve log target and places in store"""
        log_posterior = np.sum(Y * np.log(A.T + 1e-8))  # -ve of cross-entropy
        log_prior = 0
        for l in range(1, self.L+1):
            weight_term = norm.logpdf(store.get_W(l), scale=self.sigma).sum()
            bias_term = norm.logpdf(store.get_b(l), scale=self.sigma).sum()
            log_prior = log_prior + weight_term + bias_term

        U = - (log_posterior + log_prior)
        store.set_U(U)

    def sgld_initialise(self):
        """Initilaise SGLD - Stochastic Gradient Langevin Diffusion for MCMC sampling form posterior"""
        self.t = 0

    def perform_mala(self, X, Y, num_iter=1000, step_scaling=1, verbose=False):
        """Performs Metropolis-Adjusted Langevin Algorithm

            Parameters:
                X (int[][]): n x D matrix of feature flags
                Y (int[][]): n x B matrix os posterior probs
                num_iter (int): number of iterations to run
            Returns:
                acceptance_ratio (float): fraction of samples accepted
                accuracy (float): final accuracy on training set
        """
        self.n = X.shape[0]

        initial_store = self.parameters.full_copy()
        A = self._forward(X, initial_store)
        self._compute_grad_U(X, Y, initial_store)
        self._compute_log_target(initial_store, A, Y)

        num_accepted = 0

        for t in tqdm(range(0, num_iter)):
            h = step_scaling * self.anneal_step_size(t, self.n)

            final_store = initial_store.full_copy()
            final_store.langevin_iterate(h)
            A = self._forward(X, final_store)
            self._compute_grad_U(X, Y, final_store)
            self._compute_log_target(final_store, A, Y)

            log_alpha = compute_log_acceptance_prob(
                initial_store, final_store, h)

            rv = np.random.uniform(low=0.0, high=1.0)
            accepted = (np.log(rv) <= log_alpha)

            if accepted:
                num_accepted += 1
                initial_store = final_store
            else:
                pass  # initial_store not accepted

            self.store_history.append(initial_store.shallow_copy())

        acceptance_ratio = num_accepted / num_iter
        accuracy = self.accuracy(A, Y)
        if verbose:
            print("Sample accept ratio: {}%".format(acceptance_ratio * 100))
            print("Train. set accuracy: {}%".format(accuracy * 100))

        return acceptance_ratio, accuracy

    def sgld_iterate(self, X, Y, step_scaling=1):
        """Perform one iteration of sgld, returns previous cost"""
        self.n = X.shape[0]
        step_size = step_scaling * self.anneal_step_size(self.t, self.n)
        self.t += 1

        A = self._forward(X, self.parameters)
        self._compute_grad_U(X, Y, self.parameters)

        self.parameters.descend_gradient(step_size=step_size)
        self.parameters.add_gaussian_noise(std_dev=np.sqrt(2 * step_size))

        self.store_history.append(self.parameters.shallow_copy())

        A = self._forward(X, self.parameters)
        self._compute_log_target(self.parameters, A, Y)
        return self.parameters.get_U()

    def thin_samples(self, burn_in_pc=10, thinning_pc=20):
        """discard samples pre burn-in and only select keep thinning_pc of rest"""
        stop = len(self.store_history)
        start = int(stop * burn_in_pc / 100)
        step = int(100 / thinning_pc)

        self.store_history = self.store_history[start:stop:step]

    def anneal_step_size(self, t, n):
        return self.a * math.pow(self.b + t, -1 * self.gamma) / n

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
            A = self._forward(X, self.parameters)
            cost = -np.mean(Y * np.log(A.T + 1e-8))
            self._backward_cross_entropy_deriv(X, Y, self.parameters)
            self.parameters.descend_gradient(step_size=learning_rate)

            if loop % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:",
                      self.accuracy(A, Y) * 100)

            if loop % 10 == 0:
                self.costs.append(cost)

    def accuracy(self, A, Y):
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy

    def compute_mean_variances(self):
        D = self.layers_size[0]
        B = self.layers_size[1]

        param_means = np.zeros(shape=(B, D+1))
        param_std_devs = np.zeros(shape=(B, D+1))

        W_history = [store.get_W(1) for store in self.store_history]
        b_history = [store.get_b(1) for store in self.store_history]

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

    def feature_overlaps_null(self, feature_index, std_dev_multiplier, null_space):
        means = self.param_means[:, feature_index]
        std_devs = self.param_std_devs[:, feature_index]

        lower = means - (std_dev_multiplier * std_devs)
        upper = means + (std_dev_multiplier * std_devs)

        for i in range(0, len(lower)):
            if abs(lower[i]) < null_space or abs(upper[i]) < null_space:
                pass
            elif lower[i] < - null_space and upper[i] > null_space:
                pass
            else:
                return False

        return True

    #p plot helpers
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()

    def plot_U(self):
        plt.figure()
        U_arr = [store.get_U() / self.n for store in self.store_history]
        x_arr = np.arange((len(U_arr)))
        plt.plot(x_arr, U_arr)
        plt.xlabel("epochs")
        plt.ylabel("Normalised U")
        plt.title("Negative log target distribution")
        plt.show()
        return np.mean(U_arr)

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

    def plot_sampled_weights(self, feature_names, std_dev_multiplier=1, null_space=0, B_range=None, legend=False):
        """Plot the sampled weights.

            Parameters:
                feature_names (str[]): names of each feature
                std_dev_multiplier (float): how many std devs to consider when disregarding
                null_space (float): the distance from 0 we consider null
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

        discarded_features = []
        for d in range(0, D + 1):
            if self.feature_overlaps_null(d, std_dev_multiplier, null_space=null_space):
                discarded_features.append(d)
                print("Discarding feature {}: {}".format(d, feature_names[d]))

        eff_D = D + 1 - len(discarded_features)
        eff_d = 0

        width = 0.4 / eff_D
        midpoint = (eff_D - 1) / 2.0

        plt.figure()
        ax = plt.subplot(111)

        for d in range(0, D + 1):
            if d in discarded_features:
                pass
            else:
                color = plt_color(d)
                mean = self.param_means[:, d][B_range[0]: B_range[1]]
                std_dev = self.param_std_devs[:, d][B_range[0]: B_range[1]]

                x = np.arange(0, B, 1) + (width * (eff_d - midpoint))
                ax.errorbar(x=x, y=mean, color=color, fmt=".", yerr=std_dev *
                             std_dev_multiplier, label=feature_names[d])

                eff_d += 1

        block_centres = np.arange(0, B, 1)
        block_names = [str(num) for num in range(0, B)]

        plt.title(
            "Sampled softmax weightings (null threshold={})".format(null_space))
        plt.xlabel("Block index")
        plt.ylabel("Weight mean $\\pm {}\\sigma$".format(std_dev_multiplier))
        plt.grid()
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
        else:
            plt.legend([])
        plt.xticks(ticks=block_centres, labels=block_names)
        plt.show()

    def plot_block_principal_dims(self, feature_names, cutoff, std_dev_multiplier=1, B_range=None, legend=False):
        """Plots the top cutoff feature weights for each block"""

        D = self.layers_size[0]
        if B_range is None:
            B_range = (0, self.layers_size[1])

        B = B_range[1] - B_range[0]

        assert len(feature_names) == D

        feature_names.append("bias")
        self.compute_mean_variances()
        width = 0.6 / cutoff
        midpoint = (cutoff - 1) / 2.0

        plt.figure()
        ax = plt.subplot(111)
        for b in range(B_range[0], B_range[1]):

            mean = self.param_means[b, :]
            std_dev = self.param_std_devs[b, :]

            # no absolutes as want positive relation
            indices = np.argsort(mean)[::-1]
            top_indices = indices[0:cutoff]

            for i in range(0, cutoff):
                feat_index = top_indices[i]
                color = plt_color(feat_index)
                pos = b + width * (i - midpoint)
                ax.errorbar(x=pos, y=mean[feat_index], color=color, fmt=".",
                             yerr=std_dev[feat_index], label=feature_names[feat_index])

        block_centres = np.arange(0, B, 1)
        block_names = [str(num) for num in range(0, B)]

        plt.title("Top weight parameters for each block (cutoff={})".format(cutoff))
        plt.xlabel("Block index")
        plt.ylabel("Weight mean $\\pm {}\\sigma$".format(std_dev_multiplier))
        plt.grid()
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5)
        else:
            plt.legend([])
        plt.xticks(ticks=block_centres, labels=block_names)
        plt.show()

    def plot_sample_histogram(self, block_index=0, feat_index=0):
        W_history = [store.get_W(1) for store in self.store_history]
        n = len(W_history)
        values = [w[block_index, feat_index] for w in W_history]
        plt.hist(values)

        plt.title("Weight samples (n={})".format(n))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

    def plot_sample_history(self):
        W_history = [store.get_W(1) for store in self.store_history]

        B = W_history[0].shape[0]

        for b in range(0, B):
            mean = [np.mean(w[b, :]) for w in W_history]
            plt.plot(mean, label="block-{}".format(b))

        plt.title("Sampled softmax weightings")
        plt.xlabel("Sample index")
        plt.ylabel("Weight")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_step_sizes(self, T):
        t_arr = np.arange(0, T)
        sizes = [self.anneal_step_size(t, self.n) for t in t_arr]
        plt.plot(t_arr, sizes)
        plt.xlabel("iteration")
        plt.ylabel("step sizes")


    def plot_sample_matrix(self):
        self.compute_mean_variances()

        fig = plt.figure()
        ax = plt.subplot(111)

        mat = ax.matshow(self.param_means)
        ax.set_title("Sampled weight matrix mean $W$")
        ax.set_ylabel("Block index")
        ax.set_xlabel("Feature index")
        ax.xaxis.set_label_position("top")
        fig.colorbar(mat)
        
        plt.show()

def from_values_to_one_hot(y):
    y = np.array(y)
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_hot = enc.fit_transform(y.reshape(len(y), -1))
    return y_hot
