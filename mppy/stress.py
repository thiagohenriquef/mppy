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
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    import numpy as np

    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/src/kruskal.so")

    kruskal_c = c_code.kruskal_stress
    kruskal_c.argtypes = [double_pointer, double_pointer, ctypes.c_int]
    kruskal_c.restype = None

    xpp = (distance_rn.__array_interface__['data'][0]
           + np.arange(distance_rn.shape[0]) * distance_rn.strides[0]).astype(np.uintp)
    ypp = (distance_r2.__array_interface__['data'][0]
           + np.arange(distance_r2.shape[0]) * distance_r2.strides[0]).astype(np.uintp)
    instances_ = ctypes.c_int(distance_rn.shape[0])
    kruskal_c(xpp, ypp, instances_)



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
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    import numpy as np

    distance_rn = squareform(pdist(distance_rn), 'euclidean')
    distance_r2 = squareform(pdist(distance_r2), 'euclidean')

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/src/kruskal.so")

    kruskal_c = c_code.normalized_kruskal
    kruskal_c.argtypes = [double_pointer, double_pointer, ctypes.c_int]
    kruskal_c.restype = None

    xpp = (distance_rn.__array_interface__['data'][0]
           + np.arange(distance_rn.shape[0]) * distance_rn.strides[0]).astype(np.uintp)
    ypp = (distance_r2.__array_interface__['data'][0]
           + np.arange(distance_r2.shape[0]) * distance_r2.strides[0]).astype(np.uintp)
    instances_ = ctypes.c_int(distance_rn.shape[0])
    kruskal_c(xpp, ypp, instances_)