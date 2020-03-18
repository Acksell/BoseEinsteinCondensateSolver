
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
from time import sleep



class Solver:
    def __init__(self, v0, system, sigma0, tol=0.0001, ):
        self.v0 = v0
        self.v = v0
        self.system = system
        self.dim = len(v0)
        self.tol = tol
        self.sigma = sigma0

    def iterate(self):
        self.v = self.system.iterate(self.v, self.sigma)

    def solve(self):
        prev_v = self.v.copy()
        self.iterate()
        i = 1
        # if all elements are within tol break, but take into consideration that
        # fixed points can oscillate between -v and v for an eigenstate.
        while not (np.all(abs(self.v - prev_v) < self.tol) or np.all(abs(self.v + prev_v) < self.tol)):
            prev_v = self.v.copy()
            self.iterate()
            # print("Iteration",,end="\n")
            print(i, self.v[:4])
            print(i, prev_v[:4])
            i += 1
            if i >= 50000: return None
        print("Converged in %s iterations" % i, self.v, prev_v)
        lambd = sum(map(lambda w: w[0]/w[1], zip(self.system.J(self.v).dot(self.v), self.v)))/self.dim
        # print("Eigenvalue", lambd)
        # print("Eigv quotient list:", list(map(lambda w: w[0]/w[1], zip(self.system.J(self.v).dot(self.v), self.v))))
        # print("Errors", self.v - self.system.J(self.v).dot(self.v)/lambd)
        # print("v, J*v:", self.v, self.system.J(self.v).dot(self.v))
        return lambd

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from ExampleSystem import ExampleSystem
    np.random.seed(seed=321)
    v0 = np.random.random(4)
    v0 /= np.linalg.norm(v0)

    beta = 0.8
    sigma0 = -6

    system = ExampleSystem(beta=beta)
    solver = Solver(v0, system, tol=1e-13,sigma0=sigma0)
    lambd = solver.solve()
    print(lambd)



