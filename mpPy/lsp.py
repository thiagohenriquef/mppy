import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
import forceScheme
import traceback

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