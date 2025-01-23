import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

#defining domain:
a = 0
b = 1
domain_lenght = b-a

#initial conditions of a field equation
x_0 = 0.5
sigma_0 = 0.1
def Gaussian(x, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-x_0)**2/(2*sigma**2))
psi_O = Gaussian(x_0, sigma_0)
print(psi_O)
nums = np.arange(0, 1, 0.05)  # Using NumPy array
psis = Gaussian(nums, sigma_0)
print(psis)
plt.plot(psis)
plt.ylabel('some numbers')
plt.show()