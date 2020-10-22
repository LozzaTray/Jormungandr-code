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


def two_samples_mean_ll_ratio(n, m, k, l):
    p_hat = k / n
    q_hat = l / m
    r_hat = (k + l) / (n + m)
    return n * bern_kl_divergence(p_hat, r_hat) + m * bern_kl_divergence(q_hat, r_hat)
