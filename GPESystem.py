import numpy as np
import scipy.sparse as sp

from System import System


def asym_harm_osc(x,y):
    return (x*x + 1.2*y*y)/6


class GPESystem(System):
    def __init__(self, b, omega, L, N, potential=asym_harm_osc):
        self.N = N  # N^2 interior points
        self.b = b
        self.omega=omega
        self.L = L
        self.potential = potential
        
        self.grid_x = np.linspace(-L, L, self.N + 2) # includes zero
        self.grid_y = np.linspace(-L, L, self.N + 2)
        self.inner_grid_x  = self.grid_x[1:-1]
        self.inner_grid_y = self.grid_y[1:-1]
        self.dx = abs(self.grid_x[0]-self.grid_x[1])
        assert np.allclose([-L + (N+1)*self.dx], L)
        self.beta = self.b/self.dx**2

    def naive_iteration(self, v, sigma):
        # vk+1 = alpha_k*(J(v_k) - sigma*I)^-1 * v_k
        # alpha_k = 1/norm((J(v_k) - sigma*I)^-1 * v_k)
        dim = len(v)
        J_minus_sigma = self.jacobian(v) - sigma*np.eye(dim)
        J_minus_sigma_inv = np.linalg.inv(J_minus_sigma)
        psi = J_minus_sigma_inv.dot(v)
        print("mshape",psi.shape)

        alpha_k = 1/np.linalg.norm(psi)
        # beta = 1/(self.v.dot(J_minus_sigma).dot(self.v))
        v = alpha_k*psi
        print(type(v),v.shape)
        # self.sigma = self.sigma + beta
        return v # normalised

    def sparse_iteration(self, v, sigma):
        u1 = sp.linalg.spsolve(self.C(v, sigma), v)
        
        # prep for u2
        v1, v2 = np.split(v, 2) # returns the two arrays
        diagv1, diagv2 = sp.diags(v1,format='csc'), sp.diags(v2,format='csc') # refactor, use new class State instead which caches this.
        w = sp.linalg.spsolve(2*self.beta*self.B(diagv1, diagv2), v)
        u2 = u1.dot(v)*w/(1-v.dot(w))
        v = u1+u2 
        return v/np.linalg.norm(v)

    iterate = sparse_iteration

    def D2(self):
        """Operating on N-dim"""

        return (1/self.dx**2)*(
            -2*sp.diags(np.ones(self.N),format='csc')
            + sp.diags(np.ones(self.N-1),1,format='csc')
            + sp.diags(np.ones(self.N-1),-1,format='csc')
        )

    def D(self):
        """Operating on N-dim"""
        return sp.diags(np.ones(self.N-1),1,format='csc') - sp.diags(np.ones(self.N-1),-1,format='csc')

    def L_N(self): # can also use kronsum
        return sp.kron(self.D2(), sp.eye(self.N), format='csc') + sp.kron(sp.eye(self.N), self.D2(),format='csc')

    def L_phi_N(self):
        return sp.kron(sp.diags(self.inner_grid_y,format='csc'), self.D(),format='csc') - sp.kron(self.D(), sp.diags(self.inner_grid_x,format='csc'),format='csc')        

    def V(self):
        return np.array([self.potential(x, y) for y in self.inner_grid_x for x in self.inner_grid_y])

    def ReA0(self):
        return -self.L_N()/2 + sp.diags(self.V(),format='csc')

    def ImA0(self):
        return -self.omega*self.L_phi_N()

    def A0(self): # imaginary
        return self.ReA0() + 1j*self.ImA0()

    def B(self, diagv1, diagv2): # diagv1 and diagv2 should be sparse.
        squareDiags = diagv1.dot(diagv1) + diagv2.dot(diagv2)
        return sp.bmat([[squareDiags, None], [None, squareDiags]])

    def J(self, v):
        # print(type(v),v.shape)
        # v=v.reshape((1800,))
        v1, v2 = np.split(v, 2) # returns the two arrays
        # v=np.array([v1,v2])
        vnormsquared = v1.dot(v1) + v2.dot(v2)
        # normalised
        assert np.allclose(np.linalg.norm(v/np.sqrt(vnormsquared)), 1)
        matrix = 0
        matrix += sp.bmat([
            [self.ReA0(), -self.ImA0()],
            [self.ImA0(),  self.ReA0()]
        ],format='csc')
        
        diagv1, diagv2 = sp.diags(v1,format='csc'), sp.diags(v2,format='csc')
        matrix += self.beta*sp.bmat([
            [3*diagv1.dot(diagv1) + diagv2.dot(diagv2),       2*diagv1.dot(diagv2)  ],
            [  2*diagv1.dot(diagv2),            diagv1.dot(diagv1) + 3*diagv2.dot(diagv2)]
        ],format='csc')/vnormsquared
        outer=np.outer(v,v)

        matrix += (-2*self.beta/vnormsquared)*self.B(diagv1,diagv2).dot(outer)
        return matrix
    jacobian = J

    def C(self, v, sigma):
        v1, v2 = np.split(v, 2) # returns the two arrays
        diagv1, diagv2 = sp.diags(v1,format='csc'), sp.diags(v2,format='csc')
        vnormsquared = v1.dot(v1) + v2.dot(v2)
        # normalised
        assert np.allclose(np.linalg.norm(v/np.sqrt(vnormsquared)), 1)
        matrix = 0
        matrix += sp.bmat([
            [self.ReA0(), -self.ImA0()],
            [self.ImA0(),  self.ReA0()]
        ],format='csc')
        matrix += self.beta*sp.bmat([
            [3*diagv1.dot(diagv1) + diagv2.dot(diagv2),       2*diagv1.dot(diagv2)  ],
            [  2*diagv1.dot(diagv2),            diagv1.dot(diagv1) + 3*diagv2.dot(diagv2)]
        ],format='csc')/vnormsquared
        matrix -= sigma*sp.eye(len(v),format='csc')
        return matrix
   
if __name__ == "__main__":
    system = GPESystem(b=200,omega=0.85,L=1,N=2)
    # print(system.L_N())
    v1=np.array([3,4,3,4])/5
    v2=np.array([4,3,4,3])/5
    v=np.append(v1,v2)
    print(system.J(v))
    # print("dx",system.dx)
    # print(system.D2())
    # print(system.D())