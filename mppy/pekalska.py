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
    from sklearn.preprocessing import scale
    distance_matrix = squareform(pdist(inst.data_matrix))

    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances - 1, int(3.0 * np.sqrt(inst.instances)))

    Ds = inst.data_matrix[inst.subsample_indices, :]
    distance_matrix_Ds = squareform(pdist(Ds))
    if inst.sample_data is None:
        from mppy.forceScheme import force2D
        from mppy.Model.Techniques import ForceScheme
        f = ForceScheme(Ds)
        inst.sample_data = force2D(f)

    inst.sample_data = scale(inst.sample_data)
    x, residuals, rank, s = np.linalg.lstsq(distance_matrix_Ds,inst.sample_data)
    #P = np.linalg.solve(distance_matrix_Ds, inst.sample_data)
    init2D = np.zeros((inst.instances, inst.sample_data.shape[1]))
    init2D[inst.subsample_indices, :] = inst.sample_data
    for j in range(inst.instances):
        if j in inst.subsample_indices:
            continue
        else:
            init2D[j,:] = np.dot(distance_matrix[j, inst.subsample_indices], x)

    inst.initial_2D_matrix = init2D
    return init2D

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
