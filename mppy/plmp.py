import numpy as np
import scipy as sp
import matplotlib as mpl
import traceback
import mppy.force as force
from sklearn.preprocessing import scale
from mppy.model.matrix import Matrix, Reader
from mppy.model.plot import Plot
from mppy.stress import calculate_kruskal_stress

class PLMP(Matrix):
    """

    Part-Linear Multidimensional Projection

    """
    def __init__(self, matrix,
                 sample_indices = None,
                 sample_data = None,
                 dimensionality = 2):
        super().__init__(matrix)
        self.sample_indices = sample_indices
        self.sample_data = sample_data
        self.dimensionality = dimensionality


def plmp_2d(inst):
    init2D = np.zeros((inst.instances, inst.dimensionality))
    if inst.sample_indices is None:
        inst.sample_indices = np.random.randint(0, inst.instances - 1, int(1.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.sample_indices, :]

    if inst.sample_data is None:
        aux = inst.data_matrix[inst.sample_indices, :]
        f = force.ForceScheme(aux)
        inst.sample_data = force.code(f)

    if inst.sample_indices is None:
        inst.sample_indices = np.random.randint(0, inst.instances - 1, int(1.0 * np.sqrt(inst.instances)))

    Xs = inst.data_matrix[inst.sample_indices, :]

    if inst.sample_data is None:
        aux = inst.data_matrix[inst.sample_indices, :]
        f = force.ForceScheme(aux)
        inst.sample_data = force.code(f)

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
        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = PLMP(matrix)
        result = plmp_2d(inst)

        print("Stress: ", calculate_kruskal_stress(inst.data_matrix, result))

        p = Plot(result, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()
