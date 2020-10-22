import math


def pdf(x, mu=0, sigma2=1):
    """pdf for a gaussian"""
    return (1/math.sqrt(2*pi*sigma2)) * math.exp(- ((x-mu)**2) / (2*sigma2))


def cdf(x, mu=0, sigma2=1):
    """cdf for a gaussian"""