import mppy.force as force


def lsp_2d(data_matrix, sample_indices=None, sample_proj=None, n_neighbors=15, weight = 1.0):
    """
    Least Square Projection
    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param sample_indices: ndarray(x,), optional, x < m.
        The X indices used as the projection sample. If sample.indices is None, a random sample is generated.
    :param sample_proj: ndarray(x,2), optional, x<m.
        Projection of the initial sample. If sample.indices is None, this attribute is omitted.
    :param neighbors: int, optional, neighbors=15
        number of neighbors used at the neighborhood matrix
    :return:

    See also:
        Paulovich, Fernando V., et al. "Least square projection: A fast high-precision
        multidimensional projection technique and its application to document mapping."
        IEEE Transactions on Visualization and Computer Graphics 14.3 (2008): 564-575.
    """
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    import time
    import ctypes
    from numpy.ctypeslib import ndpointer
    import pathlib, site
    
    instances = data_matrix.shape[0]
    
    start_time = time.time()
    if sample_indices is None:
        #sample_indices = np.random.choice(instances, int(3.0 * (np.sqrt(instances))), replace=False)
        sample_indices = np.random.randint(0, instances-1, int(3.0 * (np.sqrt(instances))))
        sample_proj = None

    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        #sample_proj = sammon._sammon(aux)
        sample_proj = force._force(aux)
    print("Initial projection time: %f" % (time.time() - start_time))
    

    nc = sample_indices.shape[0]
    A = np.zeros((instances+nc, instances))
    Dx = squareform(pdist(data_matrix))
    neighbors = (Dx.argsort()[:,1:n_neighbors+1]).astype(np.float64)
    sample_indices = sample_indices.astype(np.int32)
    b = np.zeros((instances+nc, 2)).astype(np.float64)

    double_pointer = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    for i in range(len(site.getsitepackages())):
        path = pathlib.Path(site.getsitepackages()[i]+"/lsp.so")
        if path.is_file():
            string = site.getsitepackages()[i] + "/lsp.so"
            break

    c_code = ctypes.CDLL(string)

    lsp_c = c_code.lsp
    lsp_c.argtypes = [double_pointer, double_pointer, double_pointer, ctypes.c_void_p, double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lsp_c.restype = None

    neighbors_pp = (neighbors.__array_interface__['data'][0]
           + np.arange(neighbors.shape[0]) * neighbors.strides[0]).astype(np.uintp)
    App = (A.__array_interface__['data'][0]
           + np.arange(A.shape[0]) * A.strides[0]).astype(np.uintp)
    bpp = (b.__array_interface__['data'][0]
           + np.arange(b.shape[0]) * b.strides[0]).astype(np.uintp)
    sample_indices_p = sample_indices.ctypes.data_as(ctypes.c_void_p)
    sample_projdpp = (sample_proj.__array_interface__['data'][0]
           + np.arange(sample_proj.shape[0]) * sample_proj.strides[0]).astype(np.uintp)
    nc_ = ctypes.c_int(nc)
    n_neighbors_ = ctypes.c_int(n_neighbors)
    instances_ = ctypes.c_int(instances)
    weight_ = ctypes.c_float(weight)

    lsp_c(neighbors_pp, App, bpp, sample_indices_p, sample_projdpp, nc_, n_neighbors_, instances_, weight_)

    x, residuals, rank, s = np.linalg.lstsq(A,b)

    print("LSP: %f seconds" % (time.time() - start_time))
    return x

def _lsp_old(data_matrix, sample_indices=None, sample_proj=None, n_neighbors=15):
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    import time

    instances = data_matrix.shape[0]

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.choice(instances, int(1.0 * (np.sqrt(instances))), replace=False)
        sample_proj = None

    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = force._force(aux)
    print("Initial projection time: %f" % (time.time() - start_time))


    nc = sample_indices.shape[0]
    A = np.zeros((instances + nc, instances))
    Dx = squareform(pdist(data_matrix))
    neighbors = Dx.argsort()[:, 1:n_neighbors + 1]
    b = np.zeros((instances + nc, 2))

    for i in range(instances):
        A[i,i] = 1.0
        array_neigh = neighbors[i,:]
        A[i,array_neigh] = (-(1.0 / n_neighbors))

    count = 0
    for i in range(instances, A.shape[0]):
        A[i, sample_indices[count]] = 1.0
        count += 1

    for j in range(sample_proj.shape[0]):
        b[j+instances] = sample_proj[j]

    # solving the system Ax=B
    x, residuals, rank, s = np.linalg.lstsq(A,b)

    print("Algorithm execution: %f seconds" % (time.time() - start_time))
    return x