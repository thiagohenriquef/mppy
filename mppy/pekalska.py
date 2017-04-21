#import mppy.force as force
import mppy.sammon as sammon

def pekalska_2d(matrix, sample_indices=None, sample_proj=None):
    """
    Pekalska Sammon Approach
    :param matrix: ndarray(m,n)
        dataset in the original multidimensional space. Must be a ndarray.
    :param sample_indices: ndarray(x,), optional, x < m.
        The X indices used as the projection sample. If sample.indices is None, a random sample is generated.
    :param sample_proj: ndarray(x,2), optional, x<m.
        Projection of the initial sample. If sample.indices is None, this attribute is omitted.
    :return: ndarray(m,2)
        The 2D representation of the data.

    See also:
        Pekalska, Elzbieta, et al. "A new method of generalizing Sammon mapping
        with application to algorithm speed-up." ASCI. Vol. 99. 1999.
    """
    import numpy as np
    import scipy as sp
    from scipy.spatial.distance import pdist, squareform
    import time

    orig_matrix = matrix
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    # creating the sample distance matrix
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))

    Ds = data_matrix[sample_indices, :]
    if sample_proj is None:
        #sample_proj = force._force(Ds)
        sample_proj = sammon._sammon(Ds)

    # creating base D
    n_rows, n_cols = sample_proj.shape
    D = np.zeros((n_rows, n_rows))
    D = squareform(pdist(Ds), 'euclidean')

    # creating base Y
    Y = sample_proj.copy()

    # finding and solving V
    P, L, U = sp.linalg.lu(D)
    #L = sp.linalg.cholesky(D)
    result = np.linalg.solve(L, Y)
    V = np.transpose(result)

    # calculating the projection (Y_base = D.base.V)
    for i in range(instances):
        row = data_matrix[i, :]
        dists = np.zeros((sample_proj.shape[0]))

        for j in range(len(dists)):
            dists[j] = np.linalg.norm(row - Ds[j,:])

        matrix_2d[i,0] = np.dot(dists, V[0,:])
        matrix_2d[i,1] = np.dot(dists, V[1,:])

    print("Algorithm execution: %s seconds" % (time.time() - start_time))

    return matrix_2d
