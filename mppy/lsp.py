import numpy as np
from mppy.force import _force
from scipy.spatial.distance import squareform, pdist
from mppy.stress import calculate_kruskal_stress
import time


def lsp_2d(matrix, sample_indices=None, sample_data=None, k=15, dim=2):
    """
    Least Square Projection
    :param matrix: a high-dimensional matrix dataset
    :param sample_indices: Samples indices used as control points
    :param sample_data: data samples of original dataset
    :param neighbors: number of neighbors
    :param dim: final dimension of the projection
    :return:
    """
    orig_matrix = matrix
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1] - 1
    matrix_2d = np.random.random((instances, 2))
    #data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances-1, int(1.0 * np.sqrt(instances)))
        sample_data = None

    if sample_data is None:
        aux = data_matrix[sample_indices, :]
        sample_data = _force(aux)

    nc = sample_data.shape[0]
    A = np.zeros((instances+nc, instances))
    Dx = squareform(pdist(data_matrix))
    for i in range(instances):
        neighbors = np.argsort(Dx[i, :])[1:k + 1]
        A[i, i] = 1.0
        alphas = Dx[i, neighbors]
        if any(alphas < 1e-9):
            alphas[np.array([idx for idx, item in enumerate(alphas) if item < 1e-9])] = 1
            alphas = 0
        else:
            alphas = 1 / alphas
            alphas = alphas / np.sum(alphas)
            alphas = alphas / np.sum(alphas)

        A[i, neighbors] = -alphas


    nc = sample_indices.shape[0]
    A = np.zeros((instances+nc, instances))

    Dx = squareform(pdist(data_matrix))
    for i in range(instances):
        neighbors = np.argsort(Dx[i, :])[1:k+1]
        A[i,i] = 1.0
        alphas = Dx[i, neighbors]
        if any(alphas < 1e-9):
            alphas[np.array([idx for idx, item in enumerate(alphas) if item < 1e-9])] = 1
            alphas = 0
        else:
            alphas = 1 / alphas
            alphas = alphas / np.sum(alphas)
            alphas = alphas / np.sum(alphas)
        A[i, neighbors] = -alphas

    count = 0
    for i in range(instances, A.shape[0]):
        A[i, sample_indices[count]] = 1.0
        count = count + 1

    b = np.zeros((instances+nc, 2))

    for j in range(sample_data.shape[0]):
        b[j+instances, 0] = sample_data[j,0]
        b[j+instances, 1] = sample_data[j,1]

    X = np.linalg.inv(np.dot(A.transpose(),A))
    Y = np.dot(np.transpose(A), b)
    matrix_2d = np.dot(X,Y)

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d