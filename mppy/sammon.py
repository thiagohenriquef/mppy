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


def sammon(inst):
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import scale
    distance_matrix = squareform(pdist(inst.data_matrix))

    for i in range(inst.num_iterations):
        nr_points = distance_matrix.size
        sum_distrn, sum1, sum2, delta_pq, c = 0

        #calcular o gradiente
        projection_aux = np.zeros((inst.instances, 2))
        for j in range(inst.instances):
            projection_aux[i] = inst.initial_2D_matrix[i]

        #calcular o erro inicial
        for j in range(nr_points):
            for k in range(nr_points):
                if

    init2D = inst.initial_2D_matrix
    return init2D

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import Sammon

        r = Reader()
        file = "isolet.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = Sammon(matrix)
        bidimensional_plot = sammon(inst)

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
