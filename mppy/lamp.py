try:
    import numpy as np
    import scipy as sp
    import sklearn as sk
    import matplotlib as mpl
    import traceback
except ImportError as e:
    print("Please install the following packages: ")
    print("Numpy: http://www.numpy.org/")
    print("Scipy: https://www.scipy.org/")
    print("Scikit Learn: http://scikit-learn.org/stable/")


def lamp2D(inst):
    from mppy.Model.Techniques import ForceScheme
    from mppy.forceScheme import force2D

    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances-1, int(3.0 * np.sqrt(inst.instances)))
        inst.initial_sample = None

    if inst.initial_sample is None:
        aux = inst.data_matrix[inst.subsample_indices, :]
        f = ForceScheme(aux)
        inst.initial_sample = force2D(f)

    Xs = np.array((inst.data_matrix[inst.subsample_indices, :]))
    projection = inst.initial_2D_matrix.copy()
    for i in range(inst.instances):
        point = np.array(inst.data_matrix[i, :])

        #calculating alphas
        skip = False
        alphas = np.zeros((len(inst.initial_sample)))
        for j in range(len(inst.initial_sample)):
            dist = np.sum(sp.square(Xs[j,:] - point))
            if dist < 1e-6:
                #ponto muito perto do sample point
                projection[i, :] = inst.initial_sample[j, :]
                skip = True
                break

            alphas[j] = 1.0 / dist

        if skip is True:
            continue

        c = len(inst.initial_sample) * inst.proportion
        if c < len(inst.initial_sample):
            index = alphas[np.argsort(-temp)]
            j = c
            for j in range(len(inst.initial_sample)):
                alphas[index[j]] = 0
        alphas_sum = np.sum(alphas)

        #calculate \til{x} and \til{Y}
        #print(alphas.shape, Xs.shape, inst.initial_sample.shape, alphas_sum.shape)
        Xtil = np.dot(alphas, Xs) / alphas_sum
        Ytil = np.dot(alphas, inst.initial_sample) / alphas_sum

        #calculate \hat{X} and \hat{Y}
        Xhat = Xs
        Xhat[:, ] -= Xtil
        Yhat = inst.initial_sample
        Yhat[:,] -= Ytil

        d = np.dot(np.transpose(Xhat),Yhat)
        U, s, V = sp.linalg.svd(np.dot(np.transpose(Xhat),Yhat))
        aux = np.zeros((inst.dimensions, inst.initial_2D_matrix.shape[1]))

        for k in range(inst.initial_2D_matrix.shape[1]):
            aux[k, range(inst.initial_2D_matrix.shape[1])] = V[k]

        M = np.dot(U, aux)
        projection[i] = np.dot((point - Xtil), M) + Ytil

    inst.initial_2D_matrix = projection
    return projection

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import LAMP

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = LAMP(matrix)
        bidimensional_plot = lamp2D(inst)

        from mppy.tests.Stress import KruskalStress
        k = KruskalStress(inst)
        print(k.calculate())

        from mppy.Model.Plot import Plot
        p = Plot(bidimensional_plot, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()
