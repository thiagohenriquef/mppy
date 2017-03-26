try:
    import numpy as np
    import scipy as sp
    import sklearn as sk
    import matplotlib as mpl
    import traceback
    np.seterr(divide='ignore', invalid='ignore')
except ImportError as e:
    print("Please install the following packages: ")
    print("Numpy: http://www.numpy.org/")
    print("Scipy: https://www.scipy.org/")
    print("Scikit Learn: http://scikit-learn.org/stable/")


def lsp2d(inst):
    from scipy.spatial.distance import pdist, squareform
    from mppy.Model.Techniques import ForceScheme
    from mppy.forceScheme import force2D

    init2D = inst.initial_2D_matrix
    distance_matrix = squareform(pdist(inst.data_matrix))
    num_instances = inst.instances

    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, num_instances-1, int(0.1 * num_instances))
        inst.initial_subsample = None

    if inst.initial_subsample is None:
        aux = inst.orig_matrix[inst.subsample_indices, :]
        f = ForceScheme(aux)
        inst.initial_subsample = force2D(f)

    nc = len(inst.subsample_indices)
    A = np.zeros((num_instances + nc, num_instances))

    for i in range(num_instances):
        neighbors = np.argsort(distance_matrix[i, :])[1:inst.num_neighbors]
        A[i,i] = 1
        alphas = distance_matrix[i, neighbors]
        if(any(alphas < 1e-6)):
            alphas[alphas < 1e-6] = 1
        else:
            alphas = 1 / alphas
            alphas = alphas / np.sum(alphas)
            alphas = alphas / np.sum(alphas)
        A[i, neighbors] = alphas

    A[num_instances:nc, inst.subsample_indices[0:nc]] = 1
    b = np.zeros((num_instances+nc))

    Y = np.zeros((num_instances, inst.dimensionality))
    L = np.dot(np.transpose(A), A)
    S = np.linalg.cholesky(L)

    for i in range(inst.dimensionality):
        pass

    init2D = inst.initial_subsample
    return init2D

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import LSP

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = LSP(matrix)
        print(inst.subsample_indices)
        bidimensional_plot = lsp2d(inst)

        from mppy.Model.Plot import Plot
        p = Plot(bidimensional_plot, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()