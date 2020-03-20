import numpy as np
from scipy import sparse as sp

def backslash_iterate(matrix, initial, tolerance):
    prevsol = 0
    sol = initial
    i=0
    while np.allclose(prevsol-sol, tolerance):
        prevsol = sol 
        # linalg.solve(a,b)
        sol = sp.linalg.spsolve(matrix, prevsol)
        i+=1
        print("Solved linear system in %s iterations"%i)
    return sol

def SparseGPEIteration(system, v, sigma):
    """Solves the system (J - sigma*I)^-1 * v. See theorem 5.1"""
    u1 = backslash_iterate(system.C(v, sigma), v)

    # prep for u2
    v1, v2 = np.split(v, 2) # returns the two arrays
    diagv1, diagv2 = sp.diags(v1), sp.diags(v2) # refactor, use new class State instead which caches this.
    w = backslash_iterate(system.C(v,sigma), 2*system.beta*system.B(diagv1, diagv2).dot(v))
    
    u2 = v.dot(u1)*w/(1-v.dot(w))

    return u1+u2

if __name__ == "__main__":
