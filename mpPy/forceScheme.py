__autor__ = 'thiago'
""""Force Scheme Projection"""

import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform, cdist
import sys

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
    mpl.scatter(y.T[0], y.T[1], c = t)
    mpl.show()

def force(data, maxIter=50, tol=0.0, fraction=8.0, eps=1e-6):
    row, column = np.shape(data)
    data2 = data[:, range(column-1)]
    data = data2
    Y = np.random.random((row, 2))
    X = squareform(pdist(data))
    #X = cdist(data,data,'euclidean')
    index = np.random.permutation(row)

    #para iter=1 até k faça
    for i in range(maxIter):
        #para todo yi em Y
        for j in range(row):
            inst1 = index[j]

            #para todo yj em Y com yi != yj
            for k in range(row):
                inst2 = index[k]
                if inst1 != inst2:
                    #calcular vetor yi para yj
                    v = Y[inst2] - Y[inst1]
                    distR2 = np.hypot(v[0],v[1])
                    distR2 = tol if distR2 < tol else distR2
                    delta = (X[inst1][inst2] - distR2) / fraction
                    a = (sum(v)) / distR2

                    #mover yj em direção de vetor uma fração de delta
                    Y[inst2] += delta * a
    return Y


def main():
    try:
        #file = str(sys.argv[1])
        #data = readFile.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        values = data[:, range(b - 1)]
        pos = data[:, b - 1]
        print("Executando Force Scheme... ")
        y = force(values)
        plot(y, pos)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()