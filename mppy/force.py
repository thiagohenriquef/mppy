import numpy as np
from scipy.spatial.distance import pdist, squareform
from mppy.stress import calculate_kruskal_stress
import time


def force_2d(matrix, max_iter=50, delta_frac=8.0, eps=1e-6):
    """

    :param matrix:
    :param max_iter:
    :param delta_frac:
    :param eps:
    :return:
    """
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1] - 1
    # clusters = orig_matrix[:, dimensions]
    # matrix_2d = np.random.random((instances, 2))
    # clusters = clusters.astype(int)

    start_time = time.time()
    matrix_2d = _force(data_matrix, max_iter, delta_frac, eps)
    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d


def _force(matrix, max_iter=50, delta_frac=80, eps= 1e-6):
    """

    :param matrix:
    :param max_iter:
    :param delta_frac:
    :param eps:
    :return:
    """
    matrix_2d = np.random.random((matrix.shape[0], 2))
    distance_matrix = squareform(pdist(matrix))
    index = np.random.permutation(matrix.shape[0])
    for i in range(max_iter):
        for i in range(matrix.shape[0]):
            instance1 = index[i]
            for j in range(matrix.shape[0]):
                instance2 = index[j]

                if instance1 == instance2:
                    continue
                else:
                    x1x2 = matrix_2d[instance2, 0] - matrix_2d[instance1, 0]
                    y1y2 = matrix_2d[instance2, 1] - matrix_2d[instance1, 1]
                    dr2 = np.hypot(x1x2, y1y2)
                    # dr2 = np.sqrt((x1x2 * x1x2) + (y1y2 * y1y2))

                if dr2 < eps:
                    dr2 = eps

                drn = distance_matrix[instance1, instance2] - dr2
                delta = drn - dr2
                delta /= delta_frac

                matrix_2d[instance2, 0] += delta * (x1x2 / dr2)
                matrix_2d[instance2, 1] += delta * (y1y2 / dr2)

    return matrix_2d
