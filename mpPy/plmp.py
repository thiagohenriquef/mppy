import math
import numpy as np
import forceScheme
import sys
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import scale

def plot(y, t):
    import matplotlib.pyplot as mpl
    mpl.scatter(y.T[0], y.T[1], c = t)
    mpl.show()


def readInput(fileName):
    try:
        print("Carregando conjunto de dados ", fileName)
        data = np.loadtxt(fileName, delimiter=",")
        return data
    except IOError as e:
        print("Não foi possível abrir o arquivo", fileName)
        print(e)
        sys.exit(1)

def PLMP(data, sampleIndices=None, Ys=None, k=2):
    nRow, nCol = data.shape
    data2 = data[:, range(nCol - 1)]
    data = data2
    nRow, nCol = data.shape

    if type(data) is not np.array:
        data = np.array(data)

    nRow, nCol = data.shape
    if sampleIndices is None:
        sampleIndices = np.random.randint(0, nRow - 1, int(3*(math.sqrt(nRow))))

    Xs = np.zeros((len(sampleIndices), nCol))
    Xs = data[sampleIndices, :]
    if Ys is None:
        Ys = forceScheme.force(Xs)

    if type(Ys) is not np.array:
        Ys = np.array(Ys)

    Ynrow, Yncol = Ys.shape
    if Yncol != k:
        print("Dimensionalidade não corresponde a Ys")
        sys.exit(0)

    if len(sampleIndices) != Ynrow:
        print("Sample Indices e Y precisam ter o mesmo número de instancias")

    Ys = scale(Ys)
    P = np.zeros((nCol, k))
    A = np.dot(Xs.transpose(),Xs)
    L = cholesky(A)

    for i in range(k):
        b = np.dot(Xs.transpose(), Ys[:,i])
        P[:,i] = solve_triangular(L, solve_triangular(L, b, trans=True))

    Y = np.zeros((nRow, k))
    Y[sampleIndices, : ] = Ys
    for i in range(nRow):
        if i in sampleIndices:
            continue
        else:
            Y[i,:] = np.dot(data[i,:],P)

    #print(Y)
    #print(Y.shape)
    return Y


def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        values = data[:, range(b-1)]
        pos = data[:,b-1]
        print("Executando PLMP... ")
        y = PLMP(data)
        plot(y, pos)
        #print(y)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()