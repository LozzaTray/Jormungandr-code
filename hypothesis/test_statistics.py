"""
great website: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/ 
"""
import math


def bern_kl_divergence(p, q):
    """
    Returns the Kullback-Leibler Divergence between
    2 Bernoulli distributions with parameters p and q
    D(Bern(p) || Bern(q))
    """
    return p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q))


def two_samples_mean_ll_ratio(n, m, k, l, debug=False):
    p_hat = k / float(n)
    q_hat = l / float(m)
    r_hat = (k + l) / float(n + m)

    t = n * bern_kl_divergence(p_hat, r_hat) + m * bern_kl_divergence(q_hat, r_hat)

    if debug:
        print("Population sizes:\n n = {} , m = {}\n".format(n, m))
        print("Sampled mean of each:\n p^ = {} , q^ = {}\n".format(p_hat, q_hat))
        print("Joint mean:\n r^ = {}\n".format(r_hat))
        print("t-statistic: t = {}".format(t))

    return t
