try:
    import numpy as np
    import scipy as sp
    import matplotlib as mpl
    import traceback
    import mppy.force as force
    from scipy.spatial.distance import pdist, squareform
    from mppy.model.matrix import Matrix, Reader
    from mppy.model.plot import Plot
except ImportError as e:
    print("Please install the following packages: ")
    print("Numpy: http://www.numpy.org/")
    print("Scipy: https://www.scipy.org/")
    print("Scikit Learn: http://scikit-learn.org/stable/")

class LSP(Matrix):
    """

    Least Square Projeciton

    """

    def __init__(self, matrix,
                 sample_indices = None,
                 sample_project=None,
                 num_neighbors = 15,
                 dimensionality = 2):

        super().__init__(matrix)
        self.sample_indices = sample_indices
        self.sample_project = sample_project
        self.num_neighbors = num_neighbors
        self.dimensionality = dimensionality


def lsp2d(inst):
    init2D = inst.initial_2D_matrix

    if inst.sample_indices is None:
        inst.sample_indices = np.random.randint(0, inst.instances-1, int(0.1 * np.sqrt(inst.instances)))
        inst.sample_project = None

    if inst.sample_project is None:
        aux = inst.data_matrix[inst.sample_indices, :]
        f = force.ForceScheme(aux)
        inst.sample_project = force.code(f)

    nc = inst.sample_indices.shape[0]
    A = np.zeros((inst.instances+nc, inst.instances))

    Dx = squareform(pdist(inst.data_matrix))
    for i in range(inst.instances):
        neighbors = np.argsort(Dx[i, :])[1:inst.num_neighbors+1]
        A[i,i] = 1.0
        alphas = Dx[i, neighbors]
        if any(alphas < 1e-9):
            for j in range(len(alphas)):
                if j < 1e-9:
                    alphas = 0.0
                    alphas[j] = 1.0
        else:
            alphas = 1/alphas
            alphas = alphas / np.sum(alphas)
            alphas = alphas / np.sum(alphas)

        A[i, neighbors] = -alphas


def code():
    try:

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = LSP(matrix)
        print(inst.sample_indices)
        bidimensional_plot = lsp2d(inst)

        p = Plot(bidimensional_plot, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()