

"""
from builtins import print

__autor__ = 'thiago'
""""Force Scheme Projection"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
import traceback


def kruskal(orig_matrix, new_matrix):
    try:
        row, col = orig_matrix.shape
        num = 0.0
        den = 0.0
        #print(pdist(orig_matrix).shape)
        #print(squareform(pdist(orig_matrix)).shape)
        orig_matrix = squareform(pdist(orig_matrix))
        new_matrix = squareform(pdist(new_matrix))
        for i in range(row):
            for j in range(1, col):
                num += np.power(orig_matrix[i, j] - new_matrix[i, j], 2)
                den += np.power(new_matrix[i, j], 2)

        result = np.sqrt((num / den))

        return result
    except Exception as e:
        print(traceback.print_exc())


def readInput(fileName):
    try:
        print("Carregando conjunto de dados ", fileName)
        data = np.loadtxt(fileName, delimiter=",")
        return data
    except IOError as e:
        print("Não foi possível abrir o arquivo", fileName)
        print(e)
        sys.exit(1)


def plot(y, t):
    import matplotlib.pyplot as mpl
    mpl.scatter(y.T[0], y.T[1], c=t)
    mpl.show()


def force(data, Y=None, maxIter=50, tol=0.0, fraction=8.0, eps=1e-6):
    row, column = np.shape(data)
    data2 = data[:, range(column - 1)]
    data = data2

    if Y is None:
        Y = np.random.random((row, 2))
    #if Y is None:
    #    Y = data[:, 0:1]

    X = squareform(pdist(data))
    index = np.random.permutation(row)

    # para iter=1 até k faça
    for i in range(maxIter):
        # para todo yi em Y
        for j in range(row):
            inst1 = index[j]

            # para todo yj em Y com yi != yj
            for k in range(row):
                inst2 = index[k]
                if inst1 != inst2:
                    # calcular vetor yi para yj
                    v = Y[inst2] - Y[inst1]
                    distR2 = np.hypot(v[0], v[1])
                    distR2 = tol if distR2 < tol else distR2
                    delta = (X[inst1][inst2] - distR2) / fraction
                    a = (sum(v)) / distR2

                    # mover yj em direção de vetor uma fração de delta
                    Y[inst2] += delta * a

    return Y


def main():
    try:
        print("Executando Force Scheme... ")
        # file = str(sys.argv[1])
        # data = readFile.readInput(file)
        data = readInput("iris.data")

        a, b = data.shape
        new_matrix = data[:, range(b - 1)]
        positions = data[:, b - 1]

        random_matrix = np.random.random((a, 2))
        test = random_matrix.copy()
        new_matrix = force(new_matrix, random_matrix)

        stress = kruskal(test, new_matrix)

        plot(new_matrix, positions)
        print(stress)

    except Exception as e:
        print(traceback.print_exc())
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
"""