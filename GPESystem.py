import numpy as np
import scipy.sparse as sp

from System import System


def asym_harm_osc(x,y):
    return (x*x + 1.2*y*y)

def zero_boundary(L):
    def potential(x,y):
        return -np.cos(np.pi*x/L/2)*np.cos(np.pi*y/L/2)
    return potential

class GPESystem(System):
    def __init__(self, b, omega, L, N, potential=asym_harm_osc):
        self.N = N  # N^2 interior points
        self.dim = 2*N*N # real and imaginary part for each point
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

        self.D = self.getD()
        self.D2 = self.getD2()
        self.L_N = self.getL_N()
        self.L_phi_N = self.getL_phi_N()
        self.V = self.getV()
        self.ReA0 = self.getReA0()
        self.ImA0 = self.getImA0()
        self.A0 = self.getA0()

        self.BlockReImA = sp.bmat([
            [self.ReA0, -self.ImA0],
            [self.ImA0,  self.ReA0]
        ],format='csc')

    def naive_iteration(self, v, sigma):
        # vk+1 = alpha_k*(J(v_k) - sigma*I)^-1 * v_k
        # alpha_k = 1/norm((J(v_k) - sigma*I)^-1 * v_k)
        J_minus_sigma = self.jacobian(v) - sigma*sp.eye(self.dim)
        J_minus_sigma_inv = sp.linalg.inv(J_minus_sigma)
        psi = J_minus_sigma_inv.dot(v)

        alpha_k = 1/np.linalg.norm(psi)
        # beta = 1/(self.v.dot(J_minus_sigma).dot(self.v))
        v = alpha_k*psi
        # self.sigma = self.sigma + beta
        return v # normalised

    def sparse_iteration(self, v, sigma):
        assert np.allclose(np.linalg.norm(v), 1)
        C=self.C(v, sigma)
        print("Solving")
        u1 = sp.linalg.spsolve(C, v)
        
        # prep for u2
        diagv1, diagv2 = sp.diags(v[:self.dim//2],format='dia'), sp.diags(v[self.dim//2:],format='dia') # refactor, use new class State instead which caches this.
        print("Solving w")
        w = sp.linalg.spsolve(C, 2*self.beta*self.B(diagv1, diagv2).dot(v))
        u2 = u1.dot(v)*w/(1-v.dot(w))
        v_next = u1+u2 
        v_next /= np.linalg.norm(v_next)
        assert np.allclose(np.linalg.norm(v_next), 1)
        return v_next

    iterate = sparse_iteration

    def getD2(self):
        """Operating on N-dim"""

        return (1/self.dx**2)*(
            -2*sp.diags(np.ones(self.N),format='csc')
            + sp.diags(np.ones(self.N-1),1,format='csc')
            + sp.diags(np.ones(self.N-1),-1,format='csc')
        )

    def getD(self): 
        """Operating on N-dim"""
        return (sp.diags(np.ones(self.N-1),1,format='csc') - sp.diags(np.ones(self.N-1),-1,format='csc'))/(2*self.dx)

    def getL_N(self): # can also use kronsum
        return sp.kron(self.D2, sp.eye(self.N), format='csc') + sp.kron(sp.eye(self.N), self.D2,format='csc')

    def getL_phi_N(self):
        return sp.kron(sp.diags(self.inner_grid_y,format='csc'), self.D,format='csc') - sp.kron(self.D, sp.diags(self.inner_grid_x,format='csc'),format='csc')        

    def getV(self):
        return np.array([self.potential(x, y) for y in self.inner_grid_x for x in self.inner_grid_y])

    def getReA0(self):
        return -self.L_N/2 + sp.diags(self.V,format='csc')

    def getImA0(self):
        return -self.omega*self.L_phi_N 

    def getA0(self): # imaginary
        return self.ReA0 + 1j*self.ImA0

    def A(self, v):
        diagv1,diagv2 = sp.diags(v[:self.dim//2],format='csc'), sp.diags(v[self.dim//2:],format='csc')
        return self.BlockReImA + self.beta*self.B(diagv1, diagv2)/v.dot(v)

    def B(self, diagv1, diagv2): # diagv1 and diagv2 should be sparse.
        squareDiags = diagv1.dot(diagv1) + diagv2.dot(diagv2)
        return sp.bmat([[squareDiags, None], [None, squareDiags]], format='csc')

    def J(self, v): # NOT EFFICIENT WITH SPARSE BECAUSE OF np.outer
        # print(type(v),v.shape)
        # v=v.reshape((1800,))
        v1, v2 = np.split(v, 2) # returns the two arrays
        # v=np.array([v1,v2])
        vnormsquared = v1.dot(v1) + v2.dot(v2)
        # normalised
        assert np.allclose(np.linalg.norm(v/np.sqrt(vnormsquared)), 1)
        first_term = self.BlockReImA
        
        diagv1, diagv2 = sp.diags(v1,format='csc'), sp.diags(v2,format='csc')
        second_term = self.beta*sp.bmat([
            [3*diagv1.dot(diagv1) + diagv2.dot(diagv2),       2*diagv1.dot(diagv2)  ],
            [  2*diagv1.dot(diagv2),            diagv1.dot(diagv1) + 3*diagv2.dot(diagv2)]
        ],format='csc')/vnormsquared
        # outer=np.outer(v,v)
        length = len(v)
        vT = sp.csr_matrix(v.reshape(1,length))
        v_col= sp.csc_matrix(vT.reshape(length,1))
        # TODO make function that multiplies with sparse matrices instead of the entire J.
        # works = (-2*self.beta/vnormsquared)*self.B(diagv1,diagv2).dot(outer)/vnormsquared
        # assert np.all(np.equal(alternative, works))
        partial_third_term = (-2*self.beta/vnormsquared)*(self.B(diagv1,diagv2).dot(v_col))/vnormsquared
        def dot(X):
            result  = first_term.dot(X)
            result += second_term.dot(X)
            # more effective to compute vT.dot(X) than to multiply B*v (not sparse) with vT.
            result += partial_third_term.dot(vT.dot(X))
            return result
        return dot
    jacobian = J

    def C(self, v, sigma):
        v1, v2 = np.split(v, 2) # returns the two arrays
        diagv1, diagv2 = sp.diags(v1,format='csc'), sp.diags(v2,format='csc')
        vnormsquared = v1.dot(v1) + v2.dot(v2)
        # normalised
        assert np.allclose(vnormsquared, 1)
        matrix = 0
        matrix += self.BlockReImA
        diagv1_sq, diagv2_sq = diagv1.dot(diagv1), diagv2.dot(diagv2)
        matrix += self.beta*sp.bmat([
            [ 3*diagv1_sq + diagv2_sq,    2*diagv1.dot(diagv2) ],
            [ 2*diagv1.dot(diagv2),    diagv1_sq + 3*diagv2_sq ]
        ],format='csc')/vnormsquared
        matrix -= sigma*sp.eye(len(v),format='csc')
        return matrix
    
    def getSigma(self, v):
        hmax = 1e4
        epsilon = 2
        vT =    sp.csc_matrix(v.reshape(1,len(v)))
        v_col = sp.csr_matrix(v.reshape(len(v),1))
        A = self.A(v)
        p = self.getRayleigh(v)
        f = p*v_col - A.dot(v_col)
        print("||fk||",sp.linalg.norm(f))
        # self.f_list.append(sp.linalg.norm(f))
        print(1)
        g = self.J(v)(f)
        # outerprod = np.outer(v,v)
        print(2)
        I = sp.eye(self.dim)
        print(3)
        e = I.dot(-g + p*f)
        print(4)
        e-= v_col.dot(vT.dot(-g) + vT.dot(p*f))
        print(5)
        e+= v_col.dot((vT.dot(A) - p*vT.dot(I)).dot(f))
        print(6)
        h = np.sqrt(2*epsilon/sp.linalg.norm(e))
        if h > hmax:
            h = hmax
        sigma = p - 1/h
        print("returned")
        return sigma

    def getRayleigh(self, v):
        return v.dot(self.A(v).dot(v))/v.dot(v)

if __name__ == "__main__":
    system = GPESystem(b=200,omega=0.85,L=1,N=2)
    # print(system.L_N())
    v1=np.array([3,4,3,4])/5
    v2=np.array([4,3,4,3])/5
    v=np.append(v1,v2)
    print(system.J(v).toarray())
    # print("dx",system.dx)
    # print(system.D2())
    # print(system.D())