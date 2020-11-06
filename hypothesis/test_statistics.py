"""
great website: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/ 
"""
import math
from distributions import chi_squared, gaussian


def bern_kl_divergence(p, q):
    """
    Returns the Kullback-Leibler Divergence between
    2 Bernoulli distributions with parameters p and q
    D(Bern(p) || Bern(q))
    """
    offset = 1E-10
    p += offset
    q += offset
    return p * math.log(p / q) + (1-p) * math.log((1-p) / (1-q))


def two_samples_mean_ll_ratio(n, m, k, l, debug=False):
    p_hat = k / float(n)
    q_hat = l / float(m)
    r_hat = (k + l) / float(n + m)

    t = n * bern_kl_divergence(p_hat, r_hat) + m * \
        bern_kl_divergence(q_hat, r_hat)
    p = 1 - chi_squared.cdf(2*t, 1) # 2*t ~ chi_1^2 

    if debug:
        print("Population sizes:\n n = {} , m = {}\n".format(n, m))
        print("Sampled mean of each:\n p^ = {:.3f} , q^ = {:.3f}\n".format(p_hat, q_hat))
        print("Joint mean:\n r^ = {:.3f}\n".format(r_hat))
        print("t-statistic: t = {:.3f}".format(t))
        print("p-value: p = {:.7f}\n".format(p))

    return t, p


def students_z_test(n, m, k, l, debug=False):
    p_hat = k / float(n)
    q_hat = l / float(m)
    r_hat = (k + l) / float(n + m)

    var = r_hat * (1 - r_hat) * (1 / float(n) + 1 / float(m))
    z = (p_hat - q_hat) / math.sqrt(var)
    p = 2 * (1 - gaussian.cdf(abs(z))) 

    if debug:
        print("Population sizes:\n n = {} , m = {}\n".format(n, m))
        print("Sampled mean of each:\n p^ = {:.3f} , q^ = {:.3f}\n".format(p_hat, q_hat))
        print("Joint mean:\n r^ = {:.3f}\n".format(r_hat))
        print("z-statistic: z = {:.3f}".format(z))
        print("p-value: p = {:.7f}\n".format(p))

    return z, p
