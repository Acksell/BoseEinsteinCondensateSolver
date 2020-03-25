
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import matplotlib.pyplot as plt

from detect_num_vortices import num_vortices 


class Solver:
    def __init__(self, v0, system, sigma0, tol=0.0001, ):
        self.v0 = v0
        self.v = v0
        self.system = system
        self.dim = len(v0)
        self.tol = tol
        self.sigma = sigma0

    def iterate(self, dynamic_sigma=True):
        if dynamic_sigma:
            self.sigma = self.system.getSigma(self.v)
        print("Sigma:",self.sigma)
        self.v = self.system.iterate(self.v, self.sigma)


    def solve(self):
        prev_v = self.v.copy()
        self.iterate()
        i = 1
        # if all elements are within tol break, but take into consideration that
        # fixed points can oscillate between -v and v for an eigenstate.
        while not (np.all(abs(self.v - prev_v) < self.tol) or np.all(abs(self.v + prev_v) < self.tol)):
            print("Iteration",i,end="\n")
            prev_v = self.v.copy()
            self.iterate()
            if i%20 == 0:
                v1sol, v2sol = np.split(self.v, 2)
                vnorm = np.square(v1sol) + np.square(v2sol)
                # plt.imshow(vnorm.reshape(self.system.N,self.system.N), interpolation='none')
                image = vnorm.reshape(self.system.N,self.system.N)
                num_vortices(image)
                plt.figure()
                plt.contour(image,levels=[1e-10,1e-8,1e-6,1e-4,1e-3,10**(-2.5),1e-2,10**(-1.75),10**(-1.5),10**(-1.25)], interpolation='none')
                plt.colorbar()
                plt.show()
                plt.figure()
                plt.imshow(image, interpolation='none')
                # plt.contour(vnorm.reshape(self.system.N,self.system.N),levels=[1e-10,1e-8,1e-6,1e-4,1e-3,10**(-2.5),1e-2,10**(-1.75),10**(-1.5),10**(-1.25)], interpolation='none')
                plt.colorbar()
                plt.show()
            i += 1
            if i >= 50000: return None
        print("Converged in %s iterations" % i, self.v, prev_v)
        # lambd = sum(map(lambda w: w[0]/w[1], zip(self.system.J(self.v)(self.v), self.v)))/self.dim
        # print("Eigenvalue", lambd)
        # print("Eigv quotient list:", list(map(lambda w: w[0]/w[1], zip(self.system.J(self.v).dot(self.v), self.v))))
        # print("Errors", self.v - self.system.J(self.v).dot(self.v)/lambd)
        # print("v, J*v:", self.v, self.system.J(self.v).dot(self.v))
        # return lambd
        return

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



