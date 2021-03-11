import rpy2.robjects.packages as rpackages
import matplotlib.pyplot as plt

plt.figure()
plt.plot([1, 2, 3])
plt.show()

# import R's utility package
utils = rpackages.importr('utils')

sgmcmc = rpackages.importr("sgmcmc")
print(sgmcmc)