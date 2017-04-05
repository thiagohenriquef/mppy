import numpy as np
import traceback
from scipy.spatial.distance import pdist, squareform
from mppy.stress import calculate_kruskal_stress
from mppy.model.matrix import Matrix
from mppy.model.plot import Plot

class ForceScheme(Matrix):
    """
    Force Scheme Projection

    """

    def __init__(self, matrix,
                 max_iterations=50,
                 fraction_of_delta=8.0,
                 epsilon=1e-6):
        super().__init__(matrix)
        self.max_iterations = max_iterations
        self.fraction_of_delta = fraction_of_delta
        self.epsilon = epsilon


def code(inst):
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


def force_2d():
    try:
        from mppy.model.matrix import Matrix, Reader
        import time

        r = Reader()
        file = "iris.data"
        print("Carregando conjunto de dados ", file)

        matrix = r.reader_file(file)
        inst = ForceScheme(matrix)

        start_time = time.time()
        result = code(inst)
        print(time.time() - start_time, "seconds")

        print("Stress: ", calculate_kruskal_stress(inst.data_matrix, result))


        p = Plot(result, inst.clusters, matrix)
        p.semi_interactive_scatter_plot()
    except Exception as e:
        print(traceback.print_exc())


if __name__ == "__main__":
    force_2d()
