import mppy.sammon as sammon
import mppy.force as force


def lamp_2d(data_matrix, sample_indices=None, sample_proj=None, tol=1e-4, proportion=1):
    """
    Local affine multidimensional projection
    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param sample_indices: ndarray(x,), optional, x < m.
        The X indices used as the projection sample. If sample.indices is None, a random sample is generated.
    :param sample_proj: ndarray(x,2), optional, x<m.
        Projection of the initial sample. If sample.indices is None, this attribute is omitted.
    :param proportion: Proportion of nearest control points to be used.
    :return: A 2D representation of the data.

    See also:
        Joia, Paulo, et al. "Local affine multidimensional projection."
        IEEE Transactions on Visualization and Computer Graphics 17.12 (2011): 2563-2571.
    """
    import numpy as np
    import time

    instances = data_matrix.shape[0]
    matrix_2d = np.random.random((instances, 2))
    
    start_time = time.time()
    if sample_indices is None:
        #sample_indices = np.random.randint(0, instances - 1, int(3.0 * np.sqrt(instances)))
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

    

    for p in range(instances):
        X = data_matrix[p,:]
        AtB = np.zeros((d, r))
        Pstar = np.zeros((d))
        Qstar = np.zeros((r))
        Wsqrt = np.zeros((k))
        Psum = np.zeros((d))
        Qsum = np.zeros((r))
        W = np.zeros((k))
        Wsum = 0
        jump = False

        for i in range(0,k):
            P = sample_data[i,:]
            Q = sample_proj[i,:]

            W[i] = 0
            for j in range(0,d):
                W[i] += (X[j] - P[j]) * (X[j] - P[j])

            if W[i] < 1e-6:
                matrix_2d[p,0] = Q[0]
                matrix_2d[p,1] = Q[1]
                jump = True
                break

            W[i] = 1 / W[i]

            for j in range(0,d):
                Psum[j] = Psum[j] + P[j] * W[i]

            Qsum[0] = Qsum[0] + Q[0] * W[i]
            Qsum[1] = Qsum[1] + Q[1] * W[i]

            Wsum = Wsum + W[i]
            Wsqrt[i] = np.sqrt(W[i])

        if jump is True:
            continue

        for j in range(0,d):
            Pstar[j] = Psum[j] / Wsum

        Qstar[0] = Qsum[0] / Wsum
        Qstar[1] = Qsum[1] / Wsum


        #STEP 2
        for i in range(0,d):
            x = 0.0
            y = 0.0

            for j in range(0,k):
                P = sample_data[j,:]
                Q = sample_proj[j,:]

                aij = (P[i] - Pstar[i]) * Wsqrt[j]
                x = x + (aij * ((Q[0] - Qstar[0]) * Wsqrt[j]))
                y = y + (aij * ((Q[1] - Qstar[1]) * Wsqrt[j]))

            AtB[i,0] = x
            AtB[i,1] = y

        #STEP 3
        U, s, V = np.linalg.svd(AtB)

        x = 0
        y = 0
        for j in range(0,d):
            diff = X[j] - Pstar[j]

            x += diff * (U[j,0] * V[0,0] + U[j,1] * V[0,1])
            y += diff * (U[j,0] * V[1,0] + U[j,1] * V[1,1])

        x = x + Qstar[0]
        y = y + Qstar[1]

        matrix_2d[p,0] = x
        matrix_2d[p,1] = y


    print("LAMP: %f seconds" % (time.time() - start_time))
    return matrix_2d

def lamp_beta(x, sample_indices=None, sample_proj=None, tol=1e-4, proportion=1):
    import numpy as np
    import time

    instances, dim = x.shape
    Y = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances-1, int(3.0 * np.sqrt(instances-1)))
        sample_proj = None

    xs = x[sample_indices, :]
    if sample_proj is None:
        aux = x[sample_indices, :]
        ys = force._force(aux)
    print("Initial projection: %f seconds" % (time.time() - start_time))

    k, a = xs.shape
    ys_dim = ys.shape[1]

    for p in range(instances):
        point = x[p]
        alphas = np.zeros(k)
        skip = False

        for i in range(k):
            dist = np.sum(np.sqrt(abs(xs[i] - point)))
            if dist < tol:
                Y[p] = ys[i]
                skip = True
                break
            alphas[i] = 1.0 / dist

        if skip is True:
            continue


        xtil = np.zeros(dim)
        # computes x~ and y~ (eq 3)
        ytil = np.zeros(ys_dim)
        for i in range(k):
            xtil += alphas[i] * xs[i]
            ytil += alphas[i] * ys[i]
        xtil /= np.sum(alphas)
        ytil /= np.sum(alphas)

        A = np.zeros((k, dim))
        B = np.zeros((k, ys_dim))
        xhat = np.zeros((k, dim))
        yhat = np.zeros((k, ys_dim))
        # computation of x^ and y^ (eq 6)
        for i in range(k):
            xhat[i] = xs[i] - xtil
            yhat[i] = ys[i] - ytil
            A[i] = np.sqrt(alphas[i]) * xhat[i]
            B[i] = np.sqrt(alphas[i]) * yhat[i]

        U, D, V = np.linalg.svd(np.dot(A.T, B))  # (eq 7)
        # VV is the matrix V filled with zeros
        VV = np.zeros((dim, ys_dim))  # size of U = dim, by SVD
        for i in range(ys_dim):  # size of V = ys_dim, by SVD
            VV[i, range(ys_dim)] = V[i]

        M = np.dot(U, VV)  # (eq 7)

        Y[p] = np.dot(x[p] - xtil, M) + ytil  # (eq 8)

    print("LAMP: %f seconds" % (time.time() - start_time))
    return Y


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
    c_code = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__))+"/c_codes/lamp.so")

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