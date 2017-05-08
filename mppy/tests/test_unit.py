if __name__ == "__main__":
    import numpy as np
    import mppy
    import os

    print(path.abspath(path.join(__file__ ,"../..")),"/datasets/iris.data")
    data = np.loadtxt(path.abspath(path.join(__file__ ,"../..")),"/datasets/iris.data", delimiter=",")
    result = mppy.force_2d(data)

    print("Stress: ",mppy.normalized_kruskal_stress(data, result))

    mppy.simple_scatter_plot(result, data[:, data.shape[1]-1])