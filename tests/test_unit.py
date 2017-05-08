if __name__ == "__main__":
    import numpy as np
    import mppy
    import os

    data = np.loadtxt("/home/thiago/PycharmProjects/mppy/datasets/diabetes.data", delimiter=",")
    result = mppy.force_2d(data)

    print("Stress: ",mppy.normalized_kruskal_stress(data, result))

    mppy.simple_scatter_plot(result, data[:, data.shape[1]-1])