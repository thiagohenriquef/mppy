from mppy.force import _force


def plmp_2d(data_matrix, sample_indices=None, dim=2):
    import numpy as np
    import time

    instances, dimensions = data_matrix.shape
    initial_matrix = np.random.random((dimensions, 2))

    start_time = time.time()
    size = np.sqrt(instances) if np.sqrt(instances) > dimensions else 2.0 * np.sqrt(instances)
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(size))
        #sample_indices = np.random.choice(instances, int(size), replace=False)

    Xs = data_matrix[sample_indices, :]
    sample_data = _force(Xs)

    L = np.transpose(Xs)
    for i in range(dim):
        A = np.dot(L, np.transpose(L))
        try:
            B = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            B = np.linalg.pinv(A)
        C = np.dot(np.transpose(L), B)
        D = np.dot(np.transpose(sample_data[:, i]), C)
        initial_matrix[:, i] = D

    matrix_2d = np.zeros((instances, dim))
    for j in range(instances):
        matrix_2d[j, :] = np.dot(data_matrix[j, :], initial_matrix)

    matrix_2d[sample_indices,:] = sample_data

    print("PLMP: %f seconds" % (time.time() - start_time))
    return matrix_2d

def plmp_beta(data_matrix, sample_indices=None):
    import numpy as np
    import time
    from scipy.linalg import solve, lstsq

    instances, dimensions = data_matrix.shape
    start_time = time.time()
    size = np.sqrt(instances) if np.sqrt(instances) > dimensions else 2.0 * np.sqrt(instances)
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances-1, int(size))
        #sample_indices = np.random.choice(instances, int(size), replace=False)

    D = data_matrix[sample_indices, :]
    P = _force(data_matrix[sample_indices,:])
    try:
        aux = solve(np.dot(D.T, D), np.dot(D.T, P))
    except np.linalg.LinAlgError or RuntimeWarning:
        aux, residuals, rank,s = lstsq(np.dot(D.T, D), np.dot(D.T, P))
    projection = np.dot(data_matrix, aux)
    projection[sample_indices,:] = P

    print("PLMP beta: %f seconds" % (time.time() - start_time))
    projection = (projection - projection.min()) / (projection.max() - projection.min())
    return projection