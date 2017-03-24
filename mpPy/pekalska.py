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
    init2D = inst.initial_2D_matrix()
    distance_matrix = squareform(pdist(inst.data_matrix()))
    num_instancies = inst.instances()

    if inst.subsample_indices() is None:
        pass

    return init2D

def code():
    try:
        from mpPy.Model.Matrix import Matrix, Reader
        from mpPy.Model.Techniques import Pekalska

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = Pekalska(matrix)
        bidimensional_plot = pekalska(inst)

        from mpPy.Model.Plot import Plot
        p = Plot(bidimensional_plot, inst.clusters(), matrix)
        p.semi_interactive_scatter_plot()

    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    code()
