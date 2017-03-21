import math
import numpy as np
import forceScheme
import sys
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import lu, solve_triangular, solve, cholesky
import traceback

def scale(y, c=True, sc=True):
    x = y.copy()

    if c:
        x -= x.mean()
    if sc and c:
        x /= x.std()
    elif sc:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
    return x

def kruskal(orig_matrix, new_matrix):
    try:
        row, col = orig_matrix.shape
        num = 0.0
        den = 0.0
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
    data2 = data[:, range(nCol)]
    data = data2

    if type(data) is not np.array:
        data = np.array(data)

    nRow, nCol = data.shape
    if sampleIndices is None:
        sampleIndices = np.random.randint(0, nRow - 1, int(3*(math.sqrt(nRow))))
        #sampleIndices = np.arange(nRow)

    Xs = data[sampleIndices, :]
    #Xs = data.copy()
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

    Ys = scale(Ys, True, False)
    P = np.zeros((nCol, k))
    A = np.dot(np.transpose(Xs),Xs)
    L = cholesky(A, lower=False)

    for j in range(k):
        b = np.dot(np.transpose(Xs),Ys[:,j])
        P[:,j] = solve_triangular(L, solve_triangular(L, b, trans=1, lower=True), lower=True)

    Y = np.zeros((nRow, k))
    Y[sampleIndices, :] = Ys
    for i in range(nRow):
        if i in sampleIndices:
            continue
        else:
            Y[i,:] = np.dot(data[i,:],P)

    return Y


def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        new_matrix = data[:, range(b - 1)]
        positions = data[:, b - 1]

        random_matrix = np.random.random((a, 2))
        test = random_matrix.copy()
        new_matrix = PLMP(new_matrix)

        stress = kruskal(test, new_matrix)

        plot(new_matrix, positions)
        print(stress)

    except Exception as e:
        print(str(e))
        print(traceback.print_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()