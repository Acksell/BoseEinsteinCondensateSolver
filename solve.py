
import numpy as np
from numpy.linalg import inv
from time import sleep


a = np.array([[1., 2.], [3., 4.]])
ainv = inv(a)
np.allclose(np.dot(a, ainv), np.eye(2))



class Solver:
    def __init__(self, v0, system, tol=0.0001):
        self.v0 = v0
        self.v = v0
        self.system = system
        self.dim = len(v0)
        self.tol = tol

    def iterate(self, sigma):
        # vk+1 = alpha_k*(J(v_k) - sigma*I)^-1 * v_k
        # alpha_k = 1/norm((J(v_k) - sigma*I)^-1 * v_k)
        J_minus_sigma = self.system.jacobian(self.v) - sigma*np.eye(self.dim)
        J_minus_sigma_inv = np.linalg.inv(J_minus_sigma)
        next_v_non_normalised = J_minus_sigma_inv.dot(self.v)
        alpha_k = 1/np.linalg.norm(next_v_non_normalised)
        self.v = alpha_k*next_v_non_normalised
        return self.v # normalised

    def solve(self, sigma=None):
        if sigma is None:
            sigma = -5.7 # constant for now, later we update dynamically.
        prev_v = self.v.copy()
        self.iterate(sigma=sigma)
        i = 1
        # if all elements are within tol break, but take into consideration that
        # fixed points can oscillate between -v and v for an eigenstate.
        while not (np.all(abs(self.v - prev_v) < self.tol) or np.all(abs(self.v + prev_v) < self.tol)):
            prev_v = self.v.copy()
            self.iterate(sigma=sigma)
            print("Iteration",i)
            print("state", self.v, prev_v)
            i += 1
        print("Converged", self.v, prev_v)
        lambd = sum(map(lambda w: w[0]/w[1], zip(self.system.A(self.v).dot(self.v), self.v)))/self.dim
        print("Eigenvalue", lambd)
        print("Eigv quotient list:", list(map(lambda w: w[0]/w[1], zip(self.system.A(self.v).dot(self.v), self.v))))
        print("Errors", self.v - self.system.A(self.v).dot(self.v)/lambd)
        print("v, A*v:", self.v, self.system.A(self.v).dot(self.v))


if __name__ == "__main__":
    from ExampleSystem import ExampleSystem
    beta = 0.01
    np.random.seed(seed=321)
    v0 = np.random.random(4)

    system = ExampleSystem(beta=beta)
    solver = Solver(v0, system, tol=1e-100)
    solver.solve(sigma=-5.7)



