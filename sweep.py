import numpy as np
from matplotlib import pyplot as plt

from solve import Solver
from ExampleSystem import ExampleSystem

def sweep_sigma_iterations(beta):
    np.random.seed(seed=321)
    v0 = np.random.random(4)
    v0 /= np.linalg.norm(v0)

    sigma0s = np.linspace(-20,-0.1,1000)
    used_sigma0s = []
    iterations = []
    for sigma0 in sigma0s:            
        print(sigma0)
        system = ExampleSystem(beta=beta)
        solver = Solver(v0, system, tol=1e-10,sigma0=sigma0)
        iters = solver.solve()
        if iters is None: # dont show if didnt converge
            pass
        else:
            used_sigma0s.append(sigma0)
            iterations.append(iters)

    plt.figure()
    plt.title("Iterations to converge depending on sigma, β=%s"%beta, fontsize=16)
    plt.xlabel("$\sigma$", fontsize=14)
    # plt.yscale("log")
    plt.ylabel("Iterations",  fontsize=14)
    plt.plot(used_sigma0s, iterations, 'xr')
    plt.show()

def sweep_sigma_eigv(beta):
    np.random.seed(seed=321)
    v0 = np.random.random(4)
    v0 /= np.linalg.norm(v0)

    sigma0s = np.linspace(-14,-0.1,1000)
    lambdas = []
    for sigma0 in sigma0s:            
        print(sigma0)
        system = ExampleSystem(beta=beta)
        solver = Solver(v0, system, tol=1e-10,sigma0=sigma0)
        lambd = solver.solve()
        print(lambd)

        if lambd is None: # show as lambda = 0 if none found
            lambdas.append(0)
        else:
            lambdas.append(lambd)

    plt.figure()
    plt.title("Eigenvalue converged upon for given sigma, β=%s"%beta, fontsize=18)
    plt.xlabel("$\sigma$", fontsize=18)
    plt.ylabel("Eigenvalue",  fontsize=18)
    plt.plot(sigma0s, lambdas, 'xr')
    plt.show()

if __name__ == "__main__":
    # sweep_sigma_iterations(0.5)
    sweep_sigma_eigv(0.5)