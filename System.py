class System:
    def A(self, v):
        pass

    def jacobian(self, v):
        pass
    
    J = jacobian

    def is_eigenstate(self, v):
        # quotient between eigvect component and the new vector.
        # all entries should be close to eigenvalue
        eigv_quot = list(map(lambda w: w[0]/w[1], zip(self.J(v).dot(v), v)))
        eigval = sum(eigv_quot)/len(v) # len(v) is dimension
        print(eigv_quot, eigval)
        return np.allclose(eigv_quot, eigval, atol=1e-6)
