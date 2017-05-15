def kruskal_stress(distance_rn, distance_r2):
    """
    Kruskal Stress to measure the goodness of fit.
    :param distance_rn: ndarray(m,n)
        The original multidimensional dataset.
    :param distance_r2: ndarray(m,2)
        The lower dimension of the original dataset
    :return result: float
        The goodness of fit.
    """
    from scipy.spatial.distance import pdist, squareform
    import numpy as np
    """
    distance_r2 = squareform(pdist(distance_r2))
    distance_rn = squareform(pdist(distance_rn))
    num = np.sum(np.power(distance_r2 - distance_rn, 2))
    den = np.sum(np.power(distance_rn, 2))
    """
    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_rn.shape[0]):
            dist_rn = distance_rn[i, j]
            dist_r2 = distance_r2[i, j]

            num += (dist_rn - dist_r2) * (dist_rn - dist_r2)
            den += dist_rn * dist_rn

    result = np.sqrt(num / den)
    return result

def normalized_kruskal_stress(distance_rn, distance_r2):
    """
    Normalized Kruskal Stress to measure the goodness of fit.
    :param distance_rn: ndarray(m,n)
        The original multidimensional dataset.
    :param distance_r2: ndarray(m,2)
        The lower dimension of the original dataset
    :return result: float
        The goodness of fit, a value between 0 and 1.
    """
    from scipy.spatial.distance import pdist, squareform
    import math
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    import numpy as np

    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/c_codes/kruskal.so")

    kruskal_c = c_code.normalized_kruskal
    kruskal_c.argtypes = [double_pointer, double_pointer, ctypes.c_int]
    kruskal_c.restype = None

    xpp = (distance_rn.__array_interface__['data'][0]
           + np.arange(distance_rn.shape[0]) * distance_rn.strides[0]).astype(np.uintp)
    ypp = (distance_r2.__array_interface__['data'][0]
           + np.arange(distance_r2.shape[0]) * distance_r2.strides[0]).astype(np.uintp)
    instances_ = ctypes.c_int(distance_rn.shape[0])
    kruskal_c(xpp, ypp, instances_)

    """
    max_rn = -math.inf
    max_r2 = -math.inf

    for x in range(distance_rn.shape[0]):
        for y in range(1, distance_rn.shape[0]):
            value_rn = distance_rn[x,y]
            value_r2 = distance_r2[x,y]

            if value_r2 > max_r2:
                max_r2 = value_r2

            if value_rn > max_rn:
                max_rn = value_rn

    num = 0.0
    den = 0.0
    for i in range(distance_rn.shape[0]):
        for j in range(1, distance_rn.shape[0]):
            dist_rn = distance_rn[i, j] / max_rn
            dist_r2 = distance_r2[i, j] / max_r2

            num = num + (dist_rn - dist_r2) * (dist_rn - dist_r2)
            den = den + dist_rn * dist_rn

    result = num / den
    return result
    """