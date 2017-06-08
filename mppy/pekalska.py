import mppy.force as force

def pekalska_2d(data_matrix, sample_indices=None, sample_proj=None):
    import numpy as np
    import scipy as sp
    from scipy.spatial.distance import pdist, squareform
    import time

    instances, dimensions = data_matrix.shape
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        #sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))
        sample_indices = np.random.choice(instances, int(1.0 * np.sqrt(instances)), replace=False)

    Ds = data_matrix[sample_indices, :]
    if sample_proj is None:
        sample_proj = force._force(Ds)
    print("Initial projection time: %f" % (time.time() - start_time))

    # creating base D
    D = squareform(pdist(Ds))

    # creating base Y
    Y = sample_proj.copy()

    # finding and solving V
    result = sp.linalg.solve(D, Y)

    dists = np.zeros((sample_proj.shape[0]))
    # calculating the projection (Y_base = D.base.V)
    for i in range(instances):
        row = data_matrix[i, :]

        for j in range(len(dists)):
            dists[j] = np.linalg.norm(row - Ds[j])

        matrix_2d[i] = np.dot(dists, result)
    #matrix_2d[sample_indices] = sample_proj

    print("Pekalska Sammon's: %f seconds" % (time.time() - start_time))
    return matrix_2d
