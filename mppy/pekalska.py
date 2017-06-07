import mppy.force as force
#import mppy.sammon as sammon

def pekalska_2d(data_matrix, sample_indices=None, sample_proj=None):
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

    instances, dimensions = data_matrix.shape
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        #sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))
        sample_indices = np.random.choice(instances, int(3.0 * np.sqrt(instances)), replace=False)

    Ds = data_matrix[sample_indices, :]
    if sample_proj is None:
        sample_proj = force._force(Ds)
    print("Initial projection time: %f" % (time.time() - start_time))

    # creating base D
    D = squareform(pdist(Ds), 'euclidean')

    # creating base Y
    Y = sample_proj.copy()

    # finding and solving V
    P, L, U = sp.linalg.lu(D)
    result = sp.linalg.solve(L, Y)
    V = np.transpose(result)

    # calculating the projection (Y_base = D.base.V)
    for i in range(instances):
        row = data_matrix[i, :]
        dists = np.zeros((sample_proj.shape[0]))

        for j in range(len(dists)):
            dists[j] = np.linalg.norm(row - Ds[j,:])

        matrix_2d[i,0] = np.dot(dists, V[0,:])
        matrix_2d[i,1] = np.dot(dists, V[1,:])

    print("Pekalska Sammon's: %f seconds" % (time.time() - start_time))
    return matrix_2d
