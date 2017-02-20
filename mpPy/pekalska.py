import numpy as np
import sys
import forceScheme
import math
import sklearn as sk
import traceback

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

def pekalska(data, sampleIndices=None, Ys=None):
    try:
        if data is not np.array:
            data = np.array(data)

        nRow, nCol = data.shape
        if sampleIndices is None:
            sampleIndices = np.random.randint(0, nRow-1, 3*int(np.sqrt(nRow)))


        Ds = data[sampleIndices, :]
        if Ys is None:
            Ys = forceScheme.force(Ds)

        if Ys is not np.array:
            Ys = np.array(Ys)

        nrowY, ncolY = Ys.shape
        if len(sampleIndices) != nrowY:
            print("Erro, sample indices and Ys precisam ter a mesma quantidade de instâncias")
            sys.exit(1)

        Ys = sk.preprocessing.scale(Ys, with_std=False)
        P = np.dot(Ds.transpose(),Ys)
        Y = np.zeros((nRow, ncolY))
        Y[sampleIndices,:] = Ys


        for i in range(nRow):
            if i in sampleIndices:
                continue
            else:
                Y[i,:] = np.dot(data[i,:],P)


        return Y
    except Exception as e:
        print("Exceção")
        print(traceback.print_exc())

def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        values = data[:, range(b-1)]
        pos = data[:,b-1]
        print("Executando Pekalska... ")
        y = pekalska(values)
        plot(y, pos)
        #print(y)

    except Exception as e:
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()