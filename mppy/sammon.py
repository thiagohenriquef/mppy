def sammon(matrix, initial_projection=None, max_iter=50, magic_factor=0.3, tol=1e-4, dim=2):
    """
    Sammon Mapping.

    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param initial_projection: sample_proj: ndarray(m,dim), optional, dim<n.
        An initial configuration. If initial_projection is None, MDS will be executed.
    :param max_iter: int, optional, default is 50.
        Maximum number of iterations that the algorithm will execute.
    :param magic_factor: float, optional, default: 0.3
        Magic factor to optimize the convergence of the algorithm, originally between
        0,3 <= magic_factor <= 0,4
    :param tol: float, optional, default value is 0.0001
        Minimum distance between two points.
    :param dim: int, optional, if None, dim is 2
        The dimension of the configuration
    :return: ndarray(m, dim)
        The final configuration of the projection

    See also:
        Sammon, John W. "A nonlinear mapping for data structure analysis."
        IEEE Transactions on computers 100.5 (1969): 401-409.
    """
    import time
    start_time = time.time()
    matrix_2d = _sammon(matrix, initial_projection, max_iter, magic_factor, tol, dim)
    print("Algorithm execution: %s seconds" % (time.time() - start_time))

    return matrix_2d


def _sammon(data_matrix, initial_projection=None, max_iter=50, magic_factor=0.3, tol=1e-4, dim=2):
    """Common code for lamp_2d(), lsp_2d(), pekalska_2d(), plmp_2d and sammon()"""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from sklearn import manifold
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os

    distance_matrix = squareform(pdist(data_matrix), 'euclidean')

    if initial_projection is None:
        mds = manifold.MDS(n_components=dim, dissimilarity="euclidean")
        initial_projection = mds.fit_transform(data_matrix)

    projection_aux = initial_projection.copy()
    instances = distance_matrix.shape[0]

    """
    for i in range(max_iter):
        sum_dist_rn = 0.0

        projection_aux = initial_projection.copy()
        for a in range(distance_matrix.shape[0]):
            for b in range(distance_matrix.shape[0]):
                if distance_matrix[a, b] < tol:
                    distance_matrix[a, b] = tol

                sum_dist_rn += distance_matrix[a, b]

        c = -2 / sum_dist_rn

        for p in range(distance_matrix.shape[0]):
            for q in range(dim):
                sum_inder_1 = 0.0
                sum_inder_2 = 0.0

                for j in range(distance_matrix.shape[0]):
                    if j != p:
                        x1x2 = projection_aux[p, 0] - projection_aux[j, 0]
                        y1y2 = projection_aux[p, 1] - projection_aux[j, 1]
                        dist_pj = np.sqrt(abs(x1x2 * x1x2 + y1y2 * y1y2))

                        if dist_pj < tol:
                            dist_pj = tol

                        sum_inder_1 += ((distance_matrix[p, j] - dist_pj) /
                                        (distance_matrix[p, j] * dist_pj)) * (
                                           initial_projection[p, q] - initial_projection[j, q])

                        sum_inder_2 += (1 / (distance_matrix[p, j] * dist_pj)) * \
                                       ((distance_matrix[p, j] - dist_pj) -
                                        (
                                            (np.power((initial_projection[p, q] - initial_projection[j, q]),
                                                      2) / dist_pj) *
                                            (1 + ((distance_matrix[p, j] - dist_pj) / dist_pj))))

                delta_pq = ((c * sum_inder_1) / abs(c * sum_inder_2))
                print(delta_pq)
                initial_projection[p, q] -= magic_factor * delta_pq
    """
    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/c_codes/sammon.so")

    sammon_c = c_code.sammon
    sammon_c.argtypes = [double_pointer, double_pointer, double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
    sammon_c.restype = None

    xpp = (distance_matrix.__array_interface__['data'][0]
           + np.arange(distance_matrix.shape[0]) * distance_matrix.strides[0]).astype(np.uintp)
    ypp = (initial_projection.__array_interface__['data'][0]
           + np.arange(initial_projection.shape[0]) * initial_projection.strides[0]).astype(np.uintp)
    auxpp = (projection_aux.__array_interface__['data'][0]
           + np.arange(projection_aux.shape[0]) * projection_aux.strides[0]).astype(np.uintp)
    max_iter_ = ctypes.c_int(max_iter)
    tol_ = ctypes.c_double(tol)
    magic_factor_ = ctypes.c_double(magic_factor)
    instances_ = ctypes.c_int(instances)
    sammon_c(xpp, ypp, auxpp, instances_, max_iter_, magic_factor_, tol_)

    return initial_projection