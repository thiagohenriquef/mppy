import mppy.sammon as sammon

def  plmp_2d(matrix, sample_indices=None, dim=2):
    """
    Part Linear Multidimensional Projection
    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param sample_indices: ndarray(x,), optional
        The indices of the representative instances used as control points. If
        sample_indices is None, a random selection will be generated.
    :param dim: int, optional, default is 2
        The final target dimensionality.
    :return: ndarray(m,dim)
        The final 2D projection

    See also:
        Paulovich, Fernando V., Claudio T. Silva, and Luis G. Nonato.
        "Two-phase mapping for projecting massive data sets."
        IEEE Transactions on Visualization and Computer Graphics 16.6 (2010): 1281-1290.
    """
    import numpy as np
    import time

    orig_matrix = matrix
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1]
    initial_matrix = np.random.random((dimensions, dim))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    Xs = data_matrix[sample_indices, :]
    #if sample_data is None:
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

    return project