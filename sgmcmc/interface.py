import rpy2.robjects.packages as rpackages
import matplotlib.pyplot as plt


# import R's utility package
utils = rpackages.importr('utils')
tf = rpackages.importr('tensorflow')
tf.install_tensorflow()

try:
    sgmcmc = rpackages.importr("sgmcmc")
    print(sgmcmc)
except e:
    print(e)
