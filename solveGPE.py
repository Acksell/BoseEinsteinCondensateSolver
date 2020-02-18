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
    # for _ in range(num_gaussians):
    x0, y0 = np.random.randint(0,N), np.random.randint(0,N)
    result += gaussian_kernel(N,0.1,center=(N//2,N//2))
    return result/np.sum(result)

# gkern = gaussian_kernel(300,0.1,center=(100,100))
np.random.seed(seed=1234567)
N=30
initstate = normalised_random_gaussians(N, num_gaussians=10)
plt.figure()
plt.title("Initial state")
plt.imshow(initstate, interpolation='none')
plt.colorbar()
plt.show()


system = GPESystem(b=200, omega=0.85, L=15, N=N)
v1 = initstate.reshape(1, N*N)[0]
v2 = np.zeros((1,N*N))[0]
v = np.append(v1,v2) # vector (v1, v2)
solver = Solver(v0=v, system=system, sigma0=-10)
solver.solve()

v1sol, v2sol = np.split(solver.v, 2)

vnorm = np.square(v1sol) + np.square(v2sol)

plt.figure()
plt.imshow(vnorm.reshape(N,N), interpolation='none')
plt.colorbar()
plt.show()