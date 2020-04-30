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
        result += gaussian_kernel(N,0.1,center=(x0,y0))
    # x0, y0 = np.random.randint(0,N), np.random.randint(0,N)
    # result += gaussian_kernel(N,0.1,center=(N//2,N//2))
    return result/np.sum(result)

def state_to_image(v,N):
    v1sol, v2sol = np.split(v, 2)
    v_squared = np.square(v1sol) + np.square(v2sol)
    v_squared = v_squared.reshape(N,N)
    return v_squared

def gaussian_ring(R, N, SHOW=True):
    center = (N//2, N//2)
    x_axis = np.linspace(0, N-1, N) - center[0]
    y_axis = np.linspace(0, N-1, N) - center[1]
    res = [[np.exp(-(np.sqrt(x*x+y*y)-R)**2/R) for x in x_axis] for y in y_axis]
    res = res/np.sum(res)
    if SHOW:
        matplotlib.image.imsave("temp.png", res, cmap='afmhot')
        time.sleep(1)
        pretty_image = cv2.imread("temp.png")
        cv2.imshow("Image", pretty_image)
        cv2.waitKey(0)
    return res

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

def initialize_and_solve(b, omega, N=300, sigma0=None, SHOW=False, tol=5e-8):
    # np.random.seed(seed=1234567)
    np.random.seed(seed=123231)
    system = GPESystem(b=b, omega=omega, L=15, N=N)

    v=get_initial(N, SHOW=SHOW)
    print("Initial <psi|V|psi>:",system.get_E(v))
    solver = Solver(v0=v, system=system, sigma0=sigma0, tol=tol)
    solver.solve(dynamic_sigma=sigma0 is None, SHOW=SHOW)
    if SHOW:
        solver.plot_v()
    return solver.sigma

if __name__ == "__main__":
    import sys, time
    SHOW = "show" in sys.argv
    # sweep over omega and b values
    omegas = [0.4, 0.6, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95][::-1]
    omegas = [0.4, 0.6, 0.7, 0.75, 0.80][::-1]
    bees   = [100, 200, 400, 800][::-1]
    should_calculate = {100:lambda om: om == 0.8, 200:lambda om: om!= 0.95 and 0.85 > om > 0.75,400:lambda om:om not in (0.75, 0.85,0.9,0.95), 800:lambda om:om < 0.85}
    b,omega=100,0.6
    sigma = initialize_and_solve(b,omega,SHOW=SHOW)
    # iteration 1242 800 0.9
    with open("sigmas.txt", 'a') as f:
        f.write("%s, %s, %s\n" % (b,omega,sigma))
        time.sleep(2)

    # # b 800, om 0.8
    # for omega in omegas:
    #     for b in bees: 
    #         print("(b, OMEGA) = (", b, omega,")")
    #         if should_calculate.get(b)(omega):
    #             sigma = initialize_and_solve(b, omega, SHOW=SHOW)
    #             with open("sigmas.txt", 'a') as f:
    #                 f.write("%s, %s, %s\n" % (b,omega,sigma))
    #                 time.sleep(2)
