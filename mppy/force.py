def force_2d(X, Y=None, max_iter=100, delta_frac=8.0, eps=1e-6):
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
    print("Algorithm execution: %lf seconds" % (time.time() - start_time))

    return matrix_2d


def _force(X, Y=None, max_iter=100, delta_frac=8.0, eps=1e-6):
    """ Common code for force_2d(), lamp_2d(), lsp_2d(), pekalska_2d() and plmp_2d()."""
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    from sklearn.preprocessing import normalize
    #import time
    #start_time = time.time()

    if Y is None:
        Y = np.random.random((X.shape[0], 2))

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__))+"/c_codes/force.so")

    force_c = c_code.force
    force_c.argtypes = [double_pointer, double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
    force_c.restype = None

    distance_matrix = squareform(pdist(X))
    index = np.random.permutation(X.shape[0])
    instances = X.shape[0]

    xpp = (distance_matrix.__array_interface__['data'][0]
           + np.arange(distance_matrix.shape[0]) * distance_matrix.strides[0]).astype(np.uintp)
    ypp = (Y.__array_interface__['data'][0]
           + np.arange(Y.shape[0]) * Y.strides[0]).astype(np.uintp)
    max_iter_ = ctypes.c_int(max_iter)
    delta_frac_ = ctypes.c_double(delta_frac)
    eps_ = ctypes.c_double(eps)
    instances_ = ctypes.c_int(instances)
    force_c(xpp,ypp,instances_, max_iter_,eps_, delta_frac_)

    #print("Algorithm execution: %lf seconds" % (time.time() - start_time))
    normalized = (Y-Y.min())/(Y.max()-Y.min())
    return normalized
    #return Y