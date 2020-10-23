from scipy.stats import chi2


def pdf(x, dof=1):
    return chi2.pdf(x, dof)


def cdf(x, dof=1):
    return chi2.cdf(x, dof)