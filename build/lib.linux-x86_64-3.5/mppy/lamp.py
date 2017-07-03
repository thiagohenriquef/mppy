import mppy.force as force

def lamp_2d(data_matrix, sample_indices=None, sample_proj=None, tol=1e-4, proportion=1):
    import numpy as np
    import time

    instances, dim = data_matrix.shape
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances-1, int(1.0 * np.sqrt(instances)))
        sample_proj = None

    sample_data = data_matrix[sample_indices, :]
    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = force._force(aux)
    print("Initial projection: %f seconds" % (time.time() - start_time))

    k, a = sample_data.shape
    A = np.zeros((k, dim))
    B = np.zeros((k, 2))
    aux = np.zeros((dim, 2))
    alphas = np.zeros(k)

    for p in range(instances):
        point = data_matrix[p]
        skip = False

        for i in range(k):
            dist = np.sum(np.sqrt(abs(sample_data[i] - point)))
            if dist < tol:
                matrix_2d[p] = sample_proj[i]
                skip = True
                break
            alphas[i] = 1.0 / dist

        if skip is True:
            continue

        c = int(k * proportion)
        if (c < k):
            idx = -(np.argsort(alphas))
            for j in range(c, k):
                alphas[idx[j]] = 0

        Xtil = np.zeros((dim))
        Ytil = np.zeros((2))

        for i in range(k):
            Xtil += alphas[i] * sample_data[i]
            Ytil += alphas[i] * sample_proj[i]
        Xtil /= np.sum(alphas)
        Ytil /= np.sum(alphas)

        Xhat = sample_data - Xtil
        Yhat = sample_proj - Ytil
        for i in range(k):
            A[i] = np.sqrt(alphas[i]) * Xhat[i]
            B[i] = np.sqrt(alphas[i]) * Yhat[i]

        U, s, V = np.linalg.svd(np.dot(A.T, B))
        aux[0] = V[0]
        aux[1] = V[1]

        M = np.dot(U, aux)

        matrix_2d[p] = np.dot(point - Xtil, M) + Ytil

    print("LAMP: %f seconds" % (time.time() - start_time))
    return matrix_2d


def lamp_alpha(data_matrix, sample_indices=None, sample_proj=None, proportion=1):
    import numpy as np
    import ctypes, time, os
    from numpy.ctypeslib import ndpointer

    instances = data_matrix.shape[0]
    matrix_2d = np.random.random((instances, 2))
    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.choice(instances, int(3.0 * np.sqrt(instances)), replace=False)
        sample_proj = None

    sample_data = data_matrix[sample_indices, :]
    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = force._force(aux)
    print("Initial projection: %f seconds" % (time.time() - start_time))

    d = data_matrix.shape[1]
    k = sample_data.shape[0]
    r = sample_proj.shape[1]
    n = int(max(int(k * proportion), 1))
    AtB = np.zeros((d, r))

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__))+"/src/lamp.so")

    lamp_c = c_code.lamp
    lamp_c.argtypes = [double_pointer, double_pointer, double_pointer, double_pointer, double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lamp_c.restype = None
    
    data_matrixpp = (data_matrix.__array_interface__['data'][0]
           + np.arange(data_matrix.shape[0]) * data_matrix.strides[0]).astype(np.uintp)

    matrix_2dpp = (matrix_2d.__array_interface__['data'][0]
           + np.arange(matrix_2d.shape[0]) * matrix_2d.strides[0]).astype(np.uintp)

    AtBpp = (AtB.__array_interface__['data'][0]
           + np.arange(AtB.shape[0]) * AtB.strides[0]).astype(np.uintp)

    sample_datapp = (AtB.__array_interface__['data'][0]
           + np.arange(sample_data.shape[0]) * sample_data.strides[0]).astype(np.uintp)

    sample_projpp = (sample_proj.__array_interface__['data'][0]
           + np.arange(sample_proj.shape[0]) * sample_proj.strides[0]).astype(np.uintp)

    instances_ = ctypes.c_int(instances)
    d_ = ctypes.c_int(d)
    k_ = ctypes.c_int(k)
    r_ = ctypes.c_int(r)
    n_ = ctypes.c_int(n)

    lamp_c(data_matrixpp, matrix_2dpp, AtBpp, sample_datapp, sample_projpp, instances_, d_, k_, r_, n_)

    print("LAMP: %f seconds" % (time.time() - start_time))
    return matrix_2d