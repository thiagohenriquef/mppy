if __name__ == "__main__":
    import numpy as np
    import mppy
    from os.path import dirname, abspath
    import os


    data = np.loadtxt(dirname(os.getcwd())+"/datasets/iris.data", delimiter=",")
    result = mppy.lamp_2d(data[:, 0:data.shape[1]-1])

    print("Stress: ",mppy.normalized_kruskal_stress(data[:, 0:data.shape[1]-1], result))

    mppy.simple_scatter_plot(result, data[:, data.shape[1]-1])
    #mppy.interactive_scatter_plot(result, data, data[:, data.shape[1] - 1])