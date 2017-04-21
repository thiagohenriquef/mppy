def sammon(matrix, initial_projection=None, max_iter=50, magic_factor=0.3, tol=1e-4, dim=2):
    from mppy.stress import kruskal_stress
    import time

    orig_matrix = matrix
    data_matrix = orig_matrix.copy()

    start_time = time.time()
    matrix_2d = _sammon(matrix, initial_projection, max_iter, magic_factor, tol, dim)
    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d


def _sammon(data_matrix, initial_projection=None, max_iter=50, magic_factor=0.3, tol=1e-4, dim=2):
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from sklearn import manifold

    distance_matrix = squareform(pdist(data_matrix), 'euclidean')

    if initial_projection is None:
        mds = manifold.MDS(n_components=dim, dissimilarity="euclidean")
        initial_projection = mds.fit_transform(data_matrix)

    for i in range(max_iter):
        sum_dist_rn = 0.0

        projection_aux = initial_projection.copy()
        for a in range(distance_matrix.shape[0]):
            for b in range(distance_matrix.shape[0]):
                if distance_matrix[a,b] < tol:
                    distance_matrix[a,b] = tol

                sum_dist_rn += distance_matrix[a,b]

        c = -2 / sum_dist_rn

        for p in range(distance_matrix.shape[0]):
            for q in range(dim):
                sum_inder_1 = 0.0
                sum_inder_2 = 0.0

                for j in range(distance_matrix.shape[0]):
                    if j != p:
                        x1x2 = projection_aux[p,0] - projection_aux[j,0]
                        y1y2 = projection_aux[p,1] - projection_aux[j,1]
                        dist_pj = np.sqrt(abs(x1x2 * x1x2 + y1y2 * y1y2))

                        if dist_pj < tol:
                            dist_pj = tol

                        sum_inder_1 += ((distance_matrix[p,j] - dist_pj) /
                                        (distance_matrix[p,j] * dist_pj)) * (initial_projection[p, q] - initial_projection[j, q])

                        sum_inder_2 += (1 / (distance_matrix[p,j] * dist_pj)) * \
                                       ((distance_matrix[p,j] - dist_pj) -
                                        ((np.power((initial_projection[p, q] - initial_projection[j, q]), 2) / dist_pj) *
                                         (1 + ((distance_matrix[p,j] - dist_pj) / dist_pj))))

                delta_pq = ((c * sum_inder_1) / abs(c * sum_inder_2))
                initial_projection[p, q] -= magic_factor * delta_pq
    return initial_projection