if __name__ == "__main__":
    import numpy as np
    import mppy
    from os.path import dirname, abspath
    import os

    data = np.loadtxt(dirname(os.getcwd()) + "/datasets/wdbc.data", delimiter=",")
    result = mppy.force_2d(data)

    mppy.normalized_kruskal_stress(data, result)

    mppy.simple_scatter_plot(result, data[:, data.shape[1]-1])
    #mppy.interactive_scatter_plot(result, data, data[:, data.shape[1] - 1])