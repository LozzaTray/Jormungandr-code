import numpy as np
from scipy.stats import norm


class Store:

    def __init__(self, W={}, dW={}, A={}, Z={}, U=None):
        self.W = W
        self.dW = dW
        self.A = A
        self.Z = Z
        self.U = U

    def shallow_copy(self):
        """Creates a new store object with W, b and U copied through"""
        new_store = Store(self.W.copy())
        new_store.set_U(self.U)
        return new_store

    def full_copy(self):
        """Creates a new store object with all props copied through"""
        new_store = Store(
            self.W.copy(), self.dW.copy(),
            self.A.copy(), self.Z.copy(),
            self.U
        )
        return new_store

    # weight matrices

    def get_W(self, l):
        return self.W[l]

    def set_W(self, W_instance, l):
        self.W[l] = W_instance

    # weight derivatives
    def get_dW(self, l):
        return self.dW[l]

    def set_dW(self, dW_instance, l):
        self.dW[l] = dW_instance

    # activation output vector
    def get_A(self, l):
        return self.A[l]

    def set_A(self, A_instance, l):
        self.A[l] = A_instance

    # activation input vector
    def get_Z(self, l):
        return self.Z[l]

    def set_Z(self, Z_instance, l):
        self.Z[l] = Z_instance

    # -ve log-target
    def get_U(self):
        return self.U

    def set_U(self, U):
        self.U = U

    # utility functions
    def descend_gradient(self, step_size):
        """
        W and b params updated once in direction of decreasing gradient

            Params:
                step_size (float): step_size multiplier

            Returns:
                None: results stored in place
        """

        for l in self.W.keys():
            self.W[l] = self.W[l] - step_size * self.dW[l]

    def add_gaussian_noise(self, std_dev):
        """
        adds gaussian noise to each element of W and b

            Params:
                std_dev (float): standard deviation of noise

            Returns:
                None: results stored in place
        """

        for l in self.W.keys():
            W = self.W[l]
            self.W[l] = W + std_dev * np.random.randn(*W.shape)


    def langevin_iterate(self, h):
        """
        Performs one iteration of the langevin diffusion
            
            Parameters:
                h (float): step_size

            Returns:
                None: in place update of W and b
                
        """
        self.descend_gradient(step_size=h)
        self.add_gaussian_noise(std_dev=np.sqrt(2*h))

    
def compute_log_acceptance_prob(initial, final, h):
    final_mean = initial.full_copy()
    final_mean.descend_gradient(step_size=h)

    initial_mean = final.full_copy()
    initial_mean.descend_gradient(step_size=h)

    numerator = - final.get_U() + log_prob_transition(initial_mean, initial)
    denominator = - initial.get_U() + log_prob_transition(final_mean, final)

    return min(numerator - denominator, 0)


def log_prob_transition(initial, final):
    cum_sum = 0
    for l in initial.W.keys():
        W_initial = initial.get_W(l)
        W_final = final.get_W(l)
        cum_sum += norm.logpdf(W_final - W_initial).sum()

    return cum_sum




