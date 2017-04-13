import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
import mppy.force as force
from mppy.stress import calculate_kruskal_stress
import time


def pekalska_2d(matrix, sample_indices=None, inital_sample=None):
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1] - 1
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    Ds = data_matrix[sample_indices, :]
    if inital_sample is None:
        inital_sample = force._force(Ds)

    n_rows, n_cols = inital_sample.shape

    # criando base D
    D = np.zeros((n_rows, n_rows))
    D = squareform(pdist(Ds))

    #criando base Y
    Y = np.zeros((n_rows, 2))
    Y = inital_sample

    #encontrar o V
    P, L, U = sp.linalg.lu(D)
    #L = sp.linalg.cholesky(D)
    result = np.linalg.solve(L, Y)
    V = np.transpose(result)

    #calculando a projeção
    for i in range(instances):
        row = data_matrix[i, :]
        dists = np.zeros((inital_sample.shape[0]))

        for j in range(len(dists)):
            n1 = np.linalg.norm(row)
            n2 = np.linalg.norm(Ds[j])
            value = np.sqrt(abs(n1*n1+n2*n2-2*np.dot(n1,n2)))
            #dists[j] = np.linalg.norm(row - Ds[j])
            dists[j] = value

        matrix_2d[i,0] = np.dot(dists, V[0])
        matrix_2d[i,1] = np.dot(dists, V[1])

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d
