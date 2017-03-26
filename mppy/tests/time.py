import numpy as np
from scipy.linalg import cholesky, lu, lu_factor
from scipy.spatial.distance import squareform, pdist
x = np.random.rand(5,5)

A, P, L = (lu(x))
F = lu_factor(x)
#C = cholesky(x)
print(A)
print(P)
print(L)
print(F)
#print(C)