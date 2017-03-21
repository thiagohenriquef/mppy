import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, solve_triangular
import sys
import forceScheme
import traceback

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

def LSP(matrix_dataset, subSamples=None, Init_Config_subSamples=None, k=15, q=2):
    try:
        if matrix_dataset is not np.array:
            matrix_dataset = np.array(matrix_dataset)

        matrix_nRow, matrix_nCol = matrix_dataset.shape
        if subSamples is None:
            subSamples = np.random.randint(0, matrix_nRow - 1, int(0.1 * matrix_nRow))
            Init_Config_subSamples = None

        if Init_Config_subSamples is None:
            subSamples = np.array(subSamples)
            Init_Config_subSamples = forceScheme.force(matrix_dataset[subSamples, :])

        if type(Init_Config_subSamples) is not np.array:
            Init_Config_subSamples = np.array(Init_Config_subSamples)

        init_nRow, init_nCol = Init_Config_subSamples.shape
        if len(subSamples) != init_nRow:
            print("Erro, sample indices and Init_Config_subSamples precisam ter a mesma quantidade de instâncias")
            sys.exit(1)

        if q != init_nCol:
            print("Dimensionalidade precisa ser a mesma de Init_Config_subSamples")
            sys.exit(1)

        nc = len(subSamples)
        A = np.zeros((matrix_nRow+nc,matrix_nRow))
        Dx = squareform(pdist(matrix_dataset))

        for i in range(matrix_nRow):
            neighbors = np.argsort(Dx[i,:])[1:k+1]
            A[i,i] = 1
            alphas = Dx[i, neighbors]
            if any(alphas < 1e-6):
                #index = np.where(alphas < 1e-6)
                alphas = 0
                #alphas[index] = 1
            else:
                alphas = 1/alphas
                alphas = alphas/np.sum(alphas)
                alphas = alphas/np.sum(alphas)

            A[i,neighbors] = -(alphas)


        A[matrix_nRow:nc, subSamples[0:nc]] = 1
        b = np.zeros((matrix_nRow+nc))
        b[0:matrix_nRow-1] = 0

        Y = np.zeros((matrix_nRow, q))
        L = np.dot(A.transpose(),A)
        U = cholesky(L,lower=True)
        for j in range(q):
            b[matrix_nRow:matrix_nRow+nc] = Init_Config_subSamples[:,j]
            t = np.dot(np.transpose(A),b)
            Y[:,j] = solve_triangular(U, solve_triangular(U, t, trans=1))

        for count in range(matrix_nRow):
            if count in subSamples:
                Y[subSamples, :] = Init_Config_subSamples
            else:
                continue
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