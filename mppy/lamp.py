import numpy as np
import scipy as sp
from mppy.stress import calculate_kruskal_stress
from mppy.force import _force
import time
import math


def lamp_2d(matrix, sample_indices=None, sample_proj=None, proportion=1):
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1] - 1
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))
        sample_proj = None

    sample_data = data_matrix[sample_indices, :]
    if sample_proj is None:
        aux = data_matrix[sample_indices, :]
        sample_proj = _force(aux)

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

        #step 1: obtaining p_star and q_star
        p_sum = np.zeros((d))
        q_sum = np.zeros((r))
        w_sum = 0.0
        jump = False

        #local W
        local_w.fill(math.inf)
        for i in range(k):
            P = sample_data[i]
            Q = sample_proj[i]

            w = 0.0
            for j in range(d):
                w = w + (X[j] - P[j]) * (X[j] - P[j])

            #coincident points
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

        #step 2: obtain phat, qhat, A and B

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

        #step 3: projection

        #SVD computation

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







    """
    Xs = np.array((data_matrix[sample_indices, :]))
    for i in range(instances):
        point = np.array(data_matrix[i, :])

        #calculating alphas
        skip = False
        alphas = np.zeros((len(sample_proj)))
        for j in range(len(sample_proj)):
            dist = np.sum(sp.square(Xs[j,:] - point))
            if dist < 1e-6:
                #ponto muito perto do sample point
                matrix_2d[i, :] = sample_proj[j, :]
                skip = True
                break

            alphas[j] = 1.0 / dist

        if skip is True:
            continue

        c = len(sample_proj) * proportion
        if c < len(sample_proj):
            index = alphas[np.argsort(-temp)]
            j = c
            for j in range(len(sample_proj)):
                alphas[index[j]] = 0
        alphas_sum = np.sum(alphas)

        #calculate til{x} and til{Y}
        #print(alphas.shape, Xs.shape, matrix.sample_proj.shape, alphas_sum.shape)
        Xtil = np.dot(alphas, Xs) / alphas_sum
        Ytil = np.dot(alphas, sample_proj) / alphas_sum

        #calculate \hat{X} and \hat{Y}
        Xhat = Xs
        Xhat[:, ] -= Xtil
        Yhat = sample_proj
        Yhat[:,] -= Ytil

        d = np.dot(np.transpose(Xhat),Yhat)
        U, s, V = sp.linalg.svd(np.dot(np.transpose(Xhat),Yhat))
        #aux = np.zeros((dimensions, matrix_2d.shape[1]))
        aux = np.zeros((orig_matrix.shape[1], matrix_2d.shape[1]))

        for k in range(matrix_2d.shape[1]):
            aux[k, range(matrix_2d.shape[1])] = V[k]

        M = np.dot(U, aux)
        matrix_2d[i] = np.dot((point - Xtil), M) + Ytil
    """

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d