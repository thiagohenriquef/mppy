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


def force2D(inst):
    from scipy.spatial.distance import pdist, squareform
    init2D = inst.initial_2D_matrix
    distance_matrix = squareform(pdist(inst.data_matrix))
    num_instances = inst.instances

    index = np.random.permutation(num_instances)
    for i in range(inst.max_iterations):
        for i in range(num_instances):
            instance1 = index[i]
            for j in range(num_instances):
                instance2 = index[j]

                if instance1 == instance2:
                    continue
                else:
                    x1x2 = init2D[instance2, 0] - init2D[instance1, 0]
                    y1y2 = init2D[instance2, 1] - init2D[instance1, 1]
                    dr2 = np.hypot(x1x2, y1y2)
                    #dr2 = np.sqrt((x1x2 * x1x2) + (y1y2 * y1y2))

                if dr2 < inst.epsilon:
                    dr2 = inst.epsilon


                drn = distance_matrix[instance1,instance2] - dr2
                delta = drn - dr2
                delta /= inst.fraction_of_delta

                init2D[instance2,0] += delta * (x1x2 / dr2)
                init2D[instance2,1] += delta * (y1y2 / dr2)

    inst.initial_2D_matrix = init2D
    return init2D

def code():
    try:
        from mppy.Model.Matrix import Matrix, Reader
        from mppy.Model.Techniques import ForceScheme
        import time

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = ForceScheme(matrix)
        start_time = time.time()
        bidimensional_plot = force2D(inst)
        print(time.time() - start_time)

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
