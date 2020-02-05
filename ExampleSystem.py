import numpy as np

from System import System


class ExampleSystem(System):
    def __init__(self, beta):
        self.dim = 4
        self.A0 = np.array([
            [10,21,13,16],
            [21,-26,24,2],
            [13,24,-26,37],
            [16,2,37,-4]
        ])/10
        self.B = np.array([
            [-14,16,-4,15],
            [16,10,15,-9],
            [-4,15,16,6],
            [15,-9,6,-6]
        ])/10
        self.A1 = beta*np.array([
            [20,28,12,32],
            [28,4,14,6],
            [12,14,32,34],
            [32,6,34,16]
        ])/10

    def A(self, v):
        return self.A0 + np.sin(v.dot(self.B).dot(v)/v.dot(v))*self.A1

    def jacobian(self, v):
        v_norm = v.dot(v)
        vBv = v.dot(self.B).dot(v)
        long_parenthesis = v_norm*v.dot(self.B) - vBv*v
        return self.A(v) + 2*np.cos(vBv/v_norm)*self.A1.dot(v).dot(long_parenthesis)/v_norm**2
    J = jacobian

if __name__ == "__main__":
    system = ExampleSystem(beta=0)
    v = np.array([-0.38007511,  0.67452702,  0.44213677, -0.45284797]) # beta=1
    v = np.array([-0.02374334, -0.43912274,  0.82140125, -0.36319066]) # beta=1
    v = np.array([ 0.69065738 , 0.23493648 ,-0.32858139, -0.59985956])
    print(system.is_eigenstate(v))
