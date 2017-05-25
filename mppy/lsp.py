import mppy.sammon as sammon
import mppy.force as force


def lsp_2d(matrix, sample_indices=None, sample_proj=None, n_neighbors=15):
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
    import random
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    from scipy.linalg import cho_factor, cho_solve, cholesky
    import time

    instances = matrix.shape[0]
    data_matrix = matrix.copy()

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.choice(instances, int(1.0 * np.sqrt(instances)), replace=False)
        sample_proj = None

    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        #sample_proj = sammon._sammon(aux)
        sample_proj = force._force(aux)
    print("Initial projection time: %f" % (time.time() - start_time))
    
    # creating matrix A
    nc = sample_indices.shape[0]
    A = np.zeros((instances+nc, instances))
    Dx = squareform(pdist(data_matrix))
    neighbors = Dx.argsort()[:,1:n_neighbors+1]

    for i in range(instances):
        A[i,i] = 1.0
        array_neigh = neighbors[i,:]
        A[i,array_neigh] = (-(1.0 / n_neighbors))

    count = 0
    for i in range(instances, A.shape[0]):
        A[i, sample_indices[count]] = 1.0
        count = count + 1

    # creating matrix B
    b = np.zeros((instances+nc, 2))
    for j in range(sample_proj.shape[0]):
        #b[j+instances, 0] = sample_proj[j, 0]
        #b[j+instances, 1] = sample_proj[j, 1]
        b[j+instances] = sample_proj[j]

    # solving the system Ax=B
    AtA = np.dot(A.transpose(),A)
    inv_AtA = np.linalg.inv(AtA)
    C = np.dot(inv_AtA,np.transpose(A))
    matrix_2d = np.dot(C, b)

    print("Algorithm execution: %f seconds" % (time.time() - start_time))
    #normalized = (matrix_2d-matrix_2d.min())/(matrix_2d.max()-matrix_2d.min())
    #return normalized
    return matrix_2d
