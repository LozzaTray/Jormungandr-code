import math
from scipy.stats import norm


def pdf(x, mu=0, sigma=1):
    """pdf for a gaussian"""
    return norm.pdf(x, loc=mu, scale=sigma)


def cdf(x, mu=0, sigma=1):
    """cdf for a gaussian"""
    return norm.cdf(x, loc=mu, scale=sigma)