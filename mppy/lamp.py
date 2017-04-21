import mppy.sammon as sammon


def lamp_2d(matrix, sample_indices=None, sample_proj=None, proportion=1):
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
    import math

    orig_matrix = matrix
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))
        sample_proj = None

    sample_data = data_matrix[sample_indices, :]
    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = sammon._sammon(aux)

    d = data_matrix.shape[1]
    k = sample_data.shape[0]
    r = sample_proj.shape[1]
    n = int(max(int(k * proportion), 1))

    p_star = np.zeros((d))
    q_star = np.zeros((r))

    local_w = np.zeros((n))
    AtB = np.zeros((d, r))
    neighbors_index = np.zeros((n))

    for p in range(data_matrix.shape[0]):
        X = data_matrix[p, :]

        # Step 1: obtaining p_star and q_star
        p_sum = np.zeros((d))
        q_sum = np.zeros((r))
        w_sum = 0.0
        jump = False

        # local W
        local_w.fill(math.inf)
        for i in range(k):
            P = sample_data[i]
            Q = sample_proj[i]

            w = 0.0
            for j in range(d):
                w = w + (X[j] - P[j]) * (X[j] - P[j])

            # coincident points
            if w < 1e-6:
                matrix_2d[p] = Q
                jump = True
                break

            if w < local_w[n-1]:
                for j in range(n):
                    if local_w[j] > w:
                        for m in range(n-1, j, -1):
                            local_w[m] = local_w[m - 1]
                            neighbors_index[m] = neighbors_index[m - 1]

                        local_w[j] = w
                        neighbors_index[j] = i
                        break

        if jump is True:
            continue

        i = 0
        for i in range(n):
            P = sample_data[int(neighbors_index[i])]
            Q = sample_proj[int(neighbors_index[i])]

            local_w[i] = 1.0 / local_w[i]

            for j in range(d):
                p_sum[j] = p_sum[j] + P[j] * local_w[i]

            q_sum[0] = q_sum[0] + Q[0] * local_w[i]
            q_sum[1] = q_sum[1] + Q[1] * local_w[i]

            w_sum = w_sum + local_w[i]

        for j in range(d):
            p_star[j] = p_sum[j] / w_sum

        q_star[0] = q_sum[0] / w_sum
        q_star[1] = q_sum[1] / w_sum

        # Step 2: obtain phat, qhat, A and B
        # calculating AtB
        i = 0
        j = 0
        for i in range(d):
            x = 0.0
            y = 0.0

            for j in range(n):
                P = sample_data[int(neighbors_index[j])]
                Q = sample_proj[int(neighbors_index[j])]

                w_sqrt = np.sqrt(abs(local_w[j]))

                a_ij = (P[i] - p_star[i]) * w_sqrt

                x = x + (a_ij * ((Q[0] - q_star[0]) * w_sqrt))
                y = y + (a_ij * ((Q[1] - q_star[1]) * w_sqrt))

            AtB[i, 0] = x
            AtB[i, 1] = y

        # Step 3: projection
        # SVD computation
        U, s, V = np.linalg.svd(AtB)
        v_00 = V[0,0]
        v_01 = V[0,1]
        v_10 = V[1,0]
        v_11 = V[1,1]

        x = 0
        y = 0
        for j in range(d):
            diff = X[j] - p_star[j]
            u_j0 = U[i,0]
            u_j1 = U[i,1]

            x += diff * (u_j0 * v_00 * u_j1 * v_01)
            y += diff * (u_j0 * v_10 * u_j1 * v_11)

        x = x + q_star[0]
        y = y + q_star[1]

        matrix_2d[p,0] = x
        matrix_2d[p,1] = y

    print("Algorithm execution: %.2f seconds" % (time.time() - start_time))

    return matrix_2d