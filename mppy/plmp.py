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
    from mppy.Model.Techniques import ForceScheme
    from sklearn.preprocessing import scale
    from mppy.force import force2D

    init2D = np.zeros((inst.instances, inst.dimensionality))
    #init2D = inst.initial_2D_matrix
    if inst.sample_indices is None:
        inst.sample_indices = np.random.randint(0, inst.instances - 1, int(3.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.sample_indices, :]

    if inst.sample_data is None:
        aux = inst.data_matrix[inst.sample_indices, :]
        f = ForceScheme(aux)
        inst.sample_data = force2D(f)

    if inst.sample_indices is None:
        inst.sample_indices = np.random.randint(0, inst.instances - 1, int(3.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.sample_indices, :]

    if inst.sample_data is None:
        aux = inst.data_matrix[inst.sample_indices, :]
        f = ForceScheme(aux)
        inst.sample_data = force2D(f)

    Ys = scale(inst.sample_data)
    proj_aux = np.zeros((inst.dimensions, inst.dimensionality))
    A = np.dot(np.transpose(Xs), Xs)
    P, L, U = sp.linalg.lu(A)
    #D = sp.linalg.cholesky(A)
    # print(P, L, U)
    # print(np.linalg.eigvalsh(A))

    for i in range(inst.dimensionality):
        b = np.dot(np.transpose(Xs), Ys[:, i])
        proj_aux[:,i] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U, b, trans=1))
    init2D[inst.sample_indices, :] = Ys


    for j in range(inst.instances):
        if j in inst.sample_indices:
            continue
        else:
            init2D[j,:] = np.dot(inst.data_matrix[j,:],proj_aux)

    return init2D

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import PLMP

        r = Reader()
        file = "isolet.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = PLMP(matrix)
        bidimensional_plot = plmp_2d(inst)

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
