def force_2d(X, Y=None, max_iter=50, delta_frac=8.0, eps=1e-6):
    """
    Force Scheme Projection
    Computes Multidimensional Projection using Force-Scheme algorithm. Note that
    the projection will always be 2D in this code.

    :param X: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param Y: ndarray(m,2)
        Initial 2D configuration, if None, randomly chooses the initial configuration.
    :param max_iter: int, optional, default:50
        Maximum number of iterations that the algorithm will execute.
    :param delta_frac:  float, optional, default=8.0
        fraction to control the points movement. Larger values means less freedom to move.
    :param eps: float, optional, default value is 0.000001
        Minimum distance between two points.
    :return: ndarray(m,2)
        The bidimensional representation of the data.

    See also:
        Tejada, Eduardo, Rosane Minghim, and Luis Gustavo Nonato.
        "On improved projection techniques to support visual exploration of multi-dimensional data sets."
        Information Visualization 2.4 (2003): 218-231.
    """

    import time
    data_matrix = X.copy()

    start_time = time.time()
    matrix_2d = _force(data_matrix, Y, max_iter, delta_frac, eps)
    print("Algorithm execution: %.2f seconds" % (time.time() - start_time))

    return matrix_2d


def _force(X, Y=None, max_iter=50, delta_frac=80, eps=1e-6):
    """ Common code for force_2d(), lamp_2d(), lsp_2d(), pekalska_2d() and plmp_2d()."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    if Y is None:
        Y = np.random.random((X.shape[0], 2))

    distance_matrix = squareform(pdist(X))
    index = np.random.permutation(X.shape[0])
    for i in range(max_iter):
        for i in range(X.shape[0]):
            instance1 = index[i]
            for j in range(X.shape[0]):
                instance2 = index[j]

                if instance1 == instance2:
                    continue
                else:
                    x1x2 = Y[instance2, 0] - Y[instance1, 0]
                    y1y2 = Y[instance2, 1] - Y[instance1, 1]
                    dr2 = np.hypot(x1x2, y1y2)
                    # dr2 = np.sqrt((x1x2 * x1x2) + (y1y2 * y1y2))

                if dr2 < eps:
                    dr2 = eps

                drn = distance_matrix[instance1, instance2] - dr2
                delta = drn - dr2
                delta /= delta_frac

                Y[instance2, 0] += delta * (x1x2 / dr2)
                Y[instance2, 1] += delta * (y1y2 / dr2)

    return Y
