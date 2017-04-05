import numpy as np
import scipy as sp
import traceback
import mppy.force as force
from mppy.model.matrix import Reader, Matrix
from scipy.spatial.distance import pdist, squareform

class Pekalska(Matrix):
    """

    Pekalska Approximation

    """
    def __init__(self, matrix,
                 subsample_indices = None,
                 sample_data = None):
        super().__init__(matrix)
        self.subsample_indices = subsample_indices
        self.sample_data = sample_data


def pekalska(inst):
    #from sklearn.preprocessing import scale
    #distance_matrix = squareform(pdist(inst.data_matrix))

    if inst.subsample_indices is None:
        inst.subsample_indices = np.random.randint(0, inst.instances - 1, int(1.0 * np.sqrt(inst.instances)))

    Ds = inst.data_matrix[inst.subsample_indices, :]
    if inst.sample_data is None:
        f = force.ForceScheme(Ds)
        inst.sample_data = force.code(f)

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

        init2D[i,0] = np.dot(dists, V[0])
        init2D[i,1] = np.dot(dists, V[1])

    inst.initial_2D_matrix = init2D
    return init2D

def code():
    try:
        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = Pekalska(matrix)
        bidimensional_plot = pekalska(inst)

        from mppy import stress
        print("Stres: ", stress.calculate_kruskal_stress(matrix, bidimensional_plot))
        from mppy.model.plot import Plot
        p = Plot(bidimensional_plot, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()
