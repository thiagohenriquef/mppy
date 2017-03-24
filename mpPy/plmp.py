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


def plmp_2d(inst):
    from mpPy.Model.Techniques import ForceScheme
    from sklearn.preprocessing import scale
    from mpPy.forceScheme import force2D

    init2D = inst.initial_2D_matrix
    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances, int(3.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.subsample_indices, :]

    if inst.subsample_control_points is None:
        aux = inst.data_matrix[inst.subsample_indices, :]
        f = ForceScheme(aux)
        inst.subsample_control_points = force2D(f)

    L = np.transpose(inst.data_matrix[inst.subsample_indices,:])

    for j in range(inst.dimensionality):
        inst.initial_2D_matrix[:, j] = inst.subsample_control_points.T.dot(L.T.dot(np.linalg.inv(L.dot(L.T))))


    """
    init2D = inst.initial_2D_matrix
    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances, int(3.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.subsample_indices, :]

    if inst.subsample_control_points is None:
        aux = inst.data_matrix[inst.subsample_indices, :]
        f = ForceScheme(aux)
        inst.subsample_control_points = force2D(f)

    Ys = scale(inst.subsample_control_points)
    P = np.zeros((inst.dimensions, inst.dimensionality))
    A = np.dot(np.transpose(Xs), Xs)
    L = sp.linalg.cholesky(A)

    for i in range(inst.dimensionality):
        b = np.dot(np.transpose(Xs), Ys[:, i])
        P[:,i] = sp.linalg.solve_triangular(L, sp.linalg.solve_triangular(L, b, trans=1))

    init2D[inst.subsample_indices, : ]=Ys
    for j in range(inst.dimensionality):
        if j in inst.subsample_indices:
            continue
        else:
            init2D[j,:] = np.dot(inst.data_matrix[j,:],P)

    """
    return init2D

def code():
    try:
        from mpPy.Model.Matrix import Matrix, Reader
        from mpPy.Model.Techniques import PLMP

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = PLMP(matrix)
        bidimensional_plot = plmp_2d(inst)

        from mpPy.Model.Plot import Plot
        p = Plot(bidimensional_plot, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()
