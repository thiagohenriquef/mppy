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


def pekalska(inst):
    from scipy.spatial.distance import pdist, squareform
    #from sklearn.preprocessing import scale
    #distance_matrix = squareform(pdist(inst.data_matrix))

    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances - 1, int(1.0 * np.sqrt(inst.instances)))

    Ds = inst.data_matrix[inst.subsample_indices, :]
    if inst.sample_data is None:
        from mppy.force import force2D
        from mppy.Model.Techniques import ForceScheme
        f = ForceScheme(Ds)
        inst.sample_data = force2D(f)

    init2D = inst.initial_2D_matrix
    n_rows, n_cols = inst.sample_data.shape

    # criando base D
    D = np.zeros((n_rows, n_rows))
    D = squareform(pdist(Ds))

    #criando base Y
    Y = np.zeros((n_rows, 2))
    Y = inst.sample_data

    #encontrar o V
    P, L, U = sp.linalg.lu(D)
    result = np.linalg.solve(L, Y)
    V = np.transpose(result)

    #calculando a projeção
    for i in range(inst.instances):
        row = inst.data_matrix[i, :]
        dists = np.zeros((inst.sample_data.shape[0]))

        for j in range(len(dists)):
            dists[j] = np.linalg.norm(row - Ds[j])

        init2D[0] = np.dot(dists, V[0])
        init2D[1] = np.dot(dists, V[1])

    inst.initial_2D_matrix = init2D
    return init2D

    """
    inst.sample_data = scale(inst.sample_data)
    x, residuals, rank, s = np.linalg.lstsq(distance_matrix_Ds,inst.sample_data)
    #x = np.linalg.solve(distance_matrix_Ds, inst.sample_data)
    init2D = np.zeros((inst.instances, inst.sample_data.shape[1]))
    init2D[inst.sample_indices, :] = inst.sample_data
    for j in range(inst.instances):
        if j in inst.sample_indices:
            continue
        else:
            init2D[j,:] = np.dot(distance_matrix[j, inst.sample_indices], x)

    """

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import Pekalska

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = Pekalska(matrix)
        bidimensional_plot = pekalska(inst)

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
