import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from solve import Solver
from GPESystem import GPESystem


def gaussian_kernel(N, normstd, center=None):
    """
    Returns a single gaussian at expected center
    :param x0,y0:  the mean position (x0, y0)
    :param normstd: normstd*N is equal to the usual standard dev. sigma.
    """
    if center is  None:
        center = (N//2, N//2)
    x_axis = np.linspace(0, N-1, N) - center[0]
    y_axis = np.linspace(0, N-1, N) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(normstd*N))
    return kernel/np.sum(kernel)

def normalised_random_gaussians(N, num_gaussians):
    """Returns NxN kernel from superpositioned random 2d-gaussians."""
    result = np.zeros((N,N))
    for _ in range(num_gaussians):
        x0, y0 = np.random.randint(0,N), np.random.randint(0,N)
        result += gaussian_kernel(300,0.1,center=(x0,y0))
    return result/np.sum(result)

# gkern = gaussian_kernel(300,0.1,center=(100,100))
np.random.seed(seed=1234567)
initstate = normalised_random_gaussians(300, num_gaussians=10)

plt.figure()
plt.imshow(initstate, interpolation='none')
plt.colorbar()
plt.show()

system = GPESystem(b=200, omega=0.85, L=15, N=300)
# solver = Solver()