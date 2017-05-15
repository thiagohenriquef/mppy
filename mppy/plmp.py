import mppy.sammon as sammon
import mppy.force as force


def plmp_2d(matrix, sample_indices=None):
    """
    Part Linear Multidimensional Projection
    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param sample_indices: ndarray(x,), optional
        The indices of the representative instances used as control points. If
        sample_indices is None, a random selection will be generated.
    :return: ndarray(m,2)
        The final 2D projection

    See also:
        Paulovich, Fernando V., Claudio T. Silva, and Luis G. Nonato.
        "Two-phase mapping for projecting massive data sets."
        IEEE Transactions on Visualization and Computer Graphics 16.6 (2010): 1281-1290.
    """
    import numpy as np
    import time
    import scipy as sp
    from scipy.spatial.distance import squareform, pdist

    orig_matrix = matrix
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1]
    initial_matrix = np.zeros((instances,2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    aux = data_matrix[sample_indices, :]
    sample_data = force._force(aux)

    # creating matrix D'
    D = data_matrix[sample_indices, :]

    # creating matrix P'
    P = sample_data.copy()

    # solving to find A
    DtD = np.dot(D.T,D)
    DtP = np.dot(D.T,P)

    ch = sp.linalg.cholesky(DtD, lower=True)
    transf = sp.linalg.cho_solve((ch, True), DtP)
    #transf = sp.linalg.solve(DtD, DtP)
    aux_Ax = transf[:,0]
    aux_Ay = transf[:,1]

    # calculating the projection P = D.A
    for i in range(instances):
        row = data_matrix[i, :]
        x = np.dot(row, aux_Ax)
        y = np.dot(row, aux_Ay)
        initial_matrix[i, 0] = x
        initial_matrix[i, 1] = y

    print("Algorithm execution: %f seconds" % (time.time() - start_time))
    return initial_matrix
    #return project
