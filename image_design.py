import matplotlib
import matplotlib.image

import numpy as np

import sys
import time

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import cv2

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
        result += gaussian_kernel(N,0.1,center=(x0,y0))
    # x0, y0 = np.random.randint(0,N), np.random.randint(0,N)
    # result += gaussian_kernel(N,0.1,center=(N//2,N//2))
    return result/np.sum(result)

def state_to_image(v,N):
    v1sol, v2sol = np.split(v, 2)
    v_squared = np.square(v1sol) + np.square(v2sol)
    v_squared = v_squared.reshape(N,N)
    return v_squared

def get_initial(N, SHOW=False):
    if len(sys.argv) > 1 and ".npy" in sys.argv[1]:
        print("Loading from last state")
        v = np.load(sys.argv[1])
        # show image to see where we are starting
        if SHOW:    
            image = state_to_image(v,N)
            matplotlib.image.imsave("vortices.png", image, cmap='afmhot')
            time.sleep(1)
            pretty_image = cv2.imread("vortices.png")
            cv2.imshow("Image", pretty_image)
            cv2.waitKey(0)
        return v
    # initstate = gaussian_ring(40, N, SHOW=True)
    initstate = normalised_random_gaussians(N, num_gaussians=10)
    if SHOW:
        plt.figure()
        plt.title("Initial state")
        plt.imshow(initstate, interpolation='none')
        plt.colorbar()
        plt.show()
    v1 = initstate.reshape(1, N*N)[0]
    # v2 = np.zeros(1,N*N)
    v2 = -initstate.reshape(1,N*N)[0]
    v = np.append(v1,v2) # vector (v1, v2)
    vnorm = np.sqrt(v1.dot(v1) + v2.dot(v2))
    v/=vnorm
    assert np.allclose(np.linalg.norm(v),1)
    return v

np.random.seed(123)
# state = np.load("states/state_24vortices_098_2000d.npy")
state = np.load("last_state.npy")
# state = get_initial(300, True)
image = state_to_image(state,300)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (75,75)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1

matplotlib.image.imsave("temp.png", image, cmap='afmhot')
pretty_image = cv2.imread("temp.png")
cv2.putText(pretty_image,'b=200,Omega=0.7', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

cv2.imshow("Image", pretty_image)
cv2.waitKey(0)

