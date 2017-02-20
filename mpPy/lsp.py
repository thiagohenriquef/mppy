import numpy as np
from numpy import inf
import sys
import forceScheme
import traceback
from past.builtins.misc import raw_input
from scipy.spatial import distance
from scipy.linalg import solve_triangular, cholesky, lu
np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')

def readInput(fileName):
    try:
        print("Carregando conjunto de dados ", fileName)
        data = np.loadtxt(fileName, delimiter=",")
        return data
    except IOError as e:
        print("Não foi possível abrir o arquivo", fileName)
        print(e)
        sys.exit(1)

def LSP(data, sampleIndices=None, Ys=None, k=15, q=2):
    try:
        if data is not np.array:
            data = np.array(data)

        nrow, ncol = data.shape
        if sampleIndices is None:
            sampleIndices = np.random.randint(0, nrow - 1, int(0.1 * nrow))
            Ys = None

        if Ys is None:
            sampleIndices = np.array(sampleIndices)
            Ys = forceScheme.force(data[sampleIndices,:])

        if type(Ys) is not np.array:
            Ys = np.array(Ys)

        nrowY, ncolY = Ys.shape
        if len(sampleIndices) != nrowY:
            print("Erro, sample indices and Ys precisam ter a mesma quantidade de instâncias")
            sys.exit(1)

        if q != ncolY:
            print("Dimensionalidade precisa ser a mesma de Ys")
            sys.exit(1)

        nc = len(sampleIndices)
        A = np.array(np.zeros((nrow + nc, nrow)))

        Dx = distance.cdist(data, data, 'euclidean')
        Dx = np.array(Dx)
        #P, L, U = lu(Dx)
        #print(U)
        #print(Dx)
        #Dx = pdist(data)
        for i in range(nrow):
            neighbors = Dx[i,:].argsort()[0:k]
            A[i, i] = 1
            alphas = Dx[i, neighbors]
            if alphas.any() < 1e-6:
                index = np.where(alphas<1e-6)[1]
                alphas = 0
                alphas[index] = 1
            else:
                alphas += 1/alphas
                alphas = alphas/sum(alphas)
                #alphas = alphas/sum(alphas)

            alphas[alphas == inf ] = 0
            #print(alphas)
            #f = raw_input()

            A[i, neighbors] = -(alphas)
        print(A)
        A = np.nan_to_num(A)
        A[nrow:nc, sampleIndices[0:nc]] = 1
        b = np.zeros((nrow + nc))

        Y = np.zeros((nrow, q))
        L = np.dot(A.transpose(),A)
        P, L, U = lu(L)
        #U = np.linalg.cholesky(L)

        #print(U.transpose())
        for j in range(q):
            b[nrow:nrow+nc] = Ys[:,j]
            T = np.dot(A.transpose(),b)
            print(type(U),type(Y),type(T))
            Y[:,j] = solve_triangular(U, solve_triangular(U.transpose(), T))

        Y[sampleIndices,:] = Ys

        return Y
    except Exception as e:
        #print("Exceção")
        print(traceback.print_exc())

def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        values = data[:, b - 1]
        print("Executando Least Square Projection... ")
        y = LSP(data)
        forceScheme.plot(y, values)
        #print(y)

    except Exception as e:
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()