import numpy as np
import scipy as sp
from mppy.stress import calculate_kruskal_stress
from mppy.force import _force
import time


def lamp_2d(matrix, sample_indices=None, initial_sample=None, proportion=1):
    orig_matrix = matrix
    # data_matrix = matrix[:, :-1]
    data_matrix = orig_matrix.copy()
    instances = orig_matrix.shape[0]
    dimensions = orig_matrix.shape[1] - 1
    matrix_2d = np.random.random((instances, 2))

    start_time = time.time()
    if sample_indices is None:
        sample_indices = np.random.randint(0, instances - 1, int(1.0 * np.sqrt(instances)))
        initial_sample = None

    if initial_sample is None:
        aux = data_matrix[sample_indices, :]
        initial_sample = _force(aux)

    Xs = np.array((data_matrix[sample_indices, :]))
    for i in range(instances):
        point = np.array(data_matrix[i, :])

        #calculating alphas
        skip = False
        alphas = np.zeros((len(initial_sample)))
        for j in range(len(initial_sample)):
            dist = np.sum(sp.square(Xs[j,:] - point))
            if dist < 1e-6:
                #ponto muito perto do sample point
                matrix_2d[i, :] = initial_sample[j, :]
                skip = True
                break

            alphas[j] = 1.0 / dist

        if skip is True:
            continue

        c = len(initial_sample) * proportion
        if c < len(initial_sample):
            index = alphas[np.argsort(-temp)]
            j = c
            for j in range(len(initial_sample)):
                alphas[index[j]] = 0
        alphas_sum = np.sum(alphas)

        #calculate \til{x} and \til{Y}
        #print(alphas.shape, Xs.shape, matrix.initial_sample.shape, alphas_sum.shape)
        Xtil = np.dot(alphas, Xs) / alphas_sum
        Ytil = np.dot(alphas, initial_sample) / alphas_sum

        #calculate \hat{X} and \hat{Y}
        Xhat = Xs
        Xhat[:, ] -= Xtil
        Yhat = initial_sample
        Yhat[:,] -= Ytil

        d = np.dot(np.transpose(Xhat),Yhat)
        U, s, V = sp.linalg.svd(np.dot(np.transpose(Xhat),Yhat))
        #aux = np.zeros((dimensions, matrix_2d.shape[1]))
        aux = np.zeros((orig_matrix.shape[1], matrix_2d.shape[1]))

        for k in range(matrix_2d.shape[1]):
            aux[k, range(matrix_2d.shape[1])] = V[k]

        M = np.dot(U, aux)
        matrix_2d[i] = np.dot((point - Xtil), M) + Ytil

    print("Algorithm execution: %s seconds" % (time.time() - start_time))
    print("Stress: %s" % calculate_kruskal_stress(data_matrix, matrix_2d))

    return matrix_2d