import numpy as np
import scipy as sp
from sklearn.preprocessing import scale
import mppy.force as force
import mppy.sammon as sammon
from mppy.stress import calculate_kruskal_stress
import time


def plmp_2d(matrix, sample_indices=None, sample_data=None, dim=2):
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1]
    initial_matrix = np.random.random((dimensions, dim))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    Xs = data_matrix[sample_indices, :]
    if sample_data is None:
        aux = data_matrix[sample_indices, :]
        #sample_data = force._force(aux)
        sample_data = sammon._sammon(aux)

    L = np.transpose(Xs)
    for i in range(dim):
        A = np.dot(L, np.transpose(L))
        B = np.linalg.inv(A)
        C = np.dot(np.transpose(L), B)
        D = np.dot(np.transpose(sample_data[:, i]), C)
        initial_matrix[:, i] = D

    project = np.zeros((instances, dim))
    for j in range(instances):
        project[j, :] = np.dot(data_matrix[j, :], initial_matrix)

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, project))

    return project
