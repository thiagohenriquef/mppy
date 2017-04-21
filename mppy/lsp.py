import mppy.sammon as sammon


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

    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    from sklearn.neighbors import kneighbors_graph
    import time

    instances = matrix.shape[0]
    data_matrix = matrix.copy()

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances-1, int(1.0 * np.sqrt(instances)))
        sample_proj = None

    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = sammon._sammon(aux)

    # creating matrix A
    nc = sample_indices.shape[0]
    A = np.zeros((instances+nc, instances))
    Dx = squareform(pdist(data_matrix))
    for i in range(instances):
        neighbors = np.argsort(Dx[i, :])[1:n_neighbors + 1]
        A[i,i] = 1.0
        alphas = Dx[i, neighbors]
        if any(alphas < 1e-9):
            alphas[np.array([idx for idx, item in enumerate(alphas) if item < 1e-9])] = 1
            alphas = 0
        else:
            alphas = 1 / alphas
            alphas = alphas / np.sum(alphas)
            alphas = alphas / np.sum(alphas)
        A[i, neighbors] = -alphas

    count = 0
    for i in range(instances, A.shape[0]):
        A[i, sample_indices[count]] = 1.0
        count = count + 1

    # creating matrix B
    b = np.zeros((instances+nc, 2))
    for j in range(sample_proj.shape[0]):
        b[j+instances, 0] = sample_proj[j, 0]
        b[j+instances, 1] = sample_proj[j, 1]

    # solving the system Ax=B
    X = np.linalg.inv(np.dot(A.transpose(),A))
    Y = np.dot(np.transpose(A), b)
    matrix_2d = np.dot(X,Y)

    print("Algorithm execution: %.2f seconds" % (time.time() - start_time))

    return matrix_2d