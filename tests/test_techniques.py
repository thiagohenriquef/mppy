if __name__ == "__main__":
    import numpy as np
    import mppy
    from os.path import dirname, abspath
    import os

    #data = np.loadtxt(dirname(os.getcwd())+"/datasets/iris.data", delimiter=",")
    data = np.loadtxt("/home/thiago/TCC/Java códigos tácito/datasets/us-counties-python.data", delimiter=";")
    print("Carregado conjunto de dados")
    result = mppy.pekalska_2d(data[:, 1:data.shape[1] - 1])

    mppy.normalized_kruskal_stress(data[:, 1:data.shape[1] - 1], result)
    # print(mppy.kruskal_stress(data,result))

    mppy.simple_scatter_plot(result, data[:, data.shape[1] - 1])
    #mppy.delaunay_scatter(result, data, data[:, data.shape[1] - 1])
    # mppy.interactive_scatter_plot(result, data, data[:, data.shape[1] - 1])
