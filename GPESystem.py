import numpy as np

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

    def D2(self):
        """Operating on N-dim"""

        return (1/self.dx**2)*(
            -2*np.diag(np.ones(self.N))
            + np.diag(np.ones(self.N-1),1)
            + np.diag(np.ones(self.N-1),-1)
        )

    def D(self):
        """Operating on N-dim"""
        return np.diag(np.ones(self.N-1),1) - np.diag(np.ones(self.N-1),-1)

    def L_N(self):
        return np.kron(self.D2(), np.eye(self.N)) + np.kron(np.eye(self.N), self.D2())

    def L_phi_N(self):
        return np.kron(np.diag(self.inner_grid_y), self.D()) - np.kron(self.D(), np.diag(self.inner_grid_x))        

    def V(self):
        return np.array([self.potential(x, y) for y in self.inner_grid_x for x in self.inner_grid_y])

    def ReA0(self):
        return -self.L_N()/2 + np.diag(self.V())

    def ImA0(self):
        return -self.omega*self.L_phi_N()

    def A0(self): # imaginary
        return self.ReA0() + 1j*self.ImA0()

    def B(self, diagv1,diagv2):
        return np.block([
            [ diagv1.dot(diagv1)+ diagv2.dot(diagv2),  np.zeros((self.N**2,self.N**2))],
            [np.zeros((self.N**2,self.N**2)),    diagv1.dot(diagv1) + diagv2.dot(diagv2)]
        ])
        

    def J(self, v):
        v1, v2 = np.split(v, 2) # returns the two arrays
        # v=np.array([v1,v2])
        vnormsquared = v1.dot(v1) + v2.dot(v2)
        # normalised
        assert np.allclose(np.linalg.norm(v/np.sqrt(vnormsquared)), 1)
        first_term = np.block([
            [self.ReA0(), -self.ImA0()],
            [self.ImA0(),  self.ReA0()]
        ])
        diagv1,diagv2 = np.diag(v1), np.diag(v2)
        second_term = self.beta*np.block([
            [3*diagv1.dot(diagv1) + diagv2.dot(diagv2),       2*diagv1.dot(diagv2)  ],
            [  2*diagv1.dot(diagv2),            diagv1.dot(diagv1) + 3*diagv2.dot(diagv2)]
        ])/vnormsquared
        outer=np.outer(v,v)
        third_term = self.B(diagv1,diagv2).dot(outer)
        third_term *= -2*self.beta/vnormsquared
        return first_term + second_term + third_term
    jacobian = J
    

   
if __name__ == "__main__":
    system = GPESystem(b=200,omega=0.85,L=1,N=2)
    # print(system.L_N())
    v1=np.array([3,4,3,4])/5
    v2=np.array([4,3,4,3])/5
    print(system.J(v1,v2))
    # print("dx",system.dx)
    # print(system.D2())
    # print(system.D())