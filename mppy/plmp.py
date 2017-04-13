import numpy as np
import scipy as sp
from sklearn.preprocessing import scale
import mppy.force as force
from mppy.stress import calculate_kruskal_stress
import time


def plmp_2d(matrix, sample_indices=None, sample_data=None, dim=2):
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    # dimensions = orig_matrix.shape[1] - 1
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    Xs = data_matrix[sample_indices, :]
    if sample_data is None:
        aux = data_matrix[sample_indices, :]
        sample_data = force._force(aux)

    #Ys = scale(sample_data)
    Ys = sample_data
    # proj_aux = np.zeros((dimensions, dim))
    proj_aux = np.zeros((data_matrix.shape[1], dim))
    A = np.dot(np.transpose(Xs), Xs)
    P, L, U = sp.linalg.lu(A)
    # L = sp.linalg.cholesky(A)

    for i in range(dim):
        b = np.dot(np.transpose(Xs), Ys[:, i])
        proj_aux[:, i] = sp.linalg.solve_triangular(L, sp.linalg.solve_triangular(L, b, trans=1))
    matrix_2d[sample_indices, :] = Ys

    for j in range(instances):
        if j in sample_indices:
            continue
        else:
            matrix_2d[j, :] = np.dot(data_matrix[j, :], proj_aux)

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d
