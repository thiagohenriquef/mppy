import math
import numpy as np
import forceScheme
import sys
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import scale

def readInput(fileName):
    try:
        print("Carregando conjunto de dados ", fileName)
        data = np.loadtxt(fileName, delimiter=",")
        return data
    except IOError as e:
        print("Não foi possível abrir o arquivo", fileName)
        print(e)
        sys.exit(1)


def LAMP(data, sampleIndices=None, Ys=None, cp=1):
    if type(data) is not np.array:
        data = np.array(data)

    nRow, nCol = data.shape
    if sampleIndices is None:
        sampleIndices = np.random.randint(0, nRow - 1, int(math.sqrt(nRow)))
        Ys = None

    if Ys is None:
        sampleIndices = np.array(sampleIndices)
        Ys = forceScheme.force(data)

    if type(Ys) is not np.array:
        Ys = np.array(Ys)

    Ynrow, Yncol = Ys.shape
    if Yncol != k:
        print("Dimensionalidade não corresponde a Ys")
        sys.exit(0)

    if len(sampleIndices) != Ynrow:
        print("Sample Indices e Y precisam ter o mesmo número de instancias")


def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = rf.readInput("iris.data")
        a, b = data.shape
        values = data[:, b - 1]
        print("Executando LAMP... ")
        y = LAMP(data)
        forceScheme.plot(y, values)
        #print(y)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()