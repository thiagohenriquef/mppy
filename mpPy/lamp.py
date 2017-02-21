import numpy as np
import forceScheme
import sys
import traceback
from scipy.linalg import cholesky, solve_triangular, svd
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


def LAMP(matrix_dataset, samplesSubset=None, Initial_2D=None, ControlPoints=1):
    try:
        if type(matrix_dataset) is not np.array:
            matrix_dataset = np.array(matrix_dataset)

        matrix_nRow, matrix_nCol = matrix_dataset.shape
        if samplesSubset is None:
            samplesSubset = np.random.randint(0, matrix_nRow - 1, int(np.sqrt(matrix_nRow)))
            Initial_2D = None

        if Initial_2D is None:
            samplesSubset = np.array(samplesSubset)
            Initial_2D = forceScheme.force(matrix_dataset[samplesSubset,:])

        if type(Initial_2D) is not np.array:
            Initial_2D = np.array(Initial_2D)

        Init_2D_Row, Init_2D_Col = Initial_2D.shape
        sampleSubset_size = len(samplesSubset)
        if sampleSubset_size != Init_2D_Row:
            print("Sample Subset e Projeção Inicial precisam ter o mesmo número de instancias")


        projection = np.zeros((matrix_nRow, Init_2D_Col))
        Xs = np.array((matrix_dataset[samplesSubset,:]))
        for i in range(matrix_nRow):
            point = np.array(matrix_dataset[i, :])

            #calculando alphas
            skip = False
            alphas = np.zeros((sampleSubset_size))
            for j in range(sampleSubset_size):
                dist = np.sum(np.square(Xs[j,:]- point))
                if dist < 1e-6:
                    #ponto muito perto do sample ploint
                    projection[i,:] = Initial_2D[j,:]
                    skip = True
                    break

                alphas[j] = 1.0 / dist

            if skip is True:
                continue
            c = sampleSubset_size * ControlPoints
            if c < sampleSubset_size:
                index = alphas[np.argsort(-temp)]
                j = c
                for j in range(sampleSubset_size):
                    alphas[index[j]] = 0

            alphas_sum = np.sum(alphas)
            alphas_sqrt = np.sqrt(alphas)

            #calculate \tilde{X} and \tilde{Y}
            Xtil = np.dot(alphas,Xs) / alphas_sum
            Ytil = np.dot(alphas,Initial_2D) / alphas_sum

            #// calculate \hat{X} and \hat{Y}
            Xhat = Xs;
            Xhat[:,] -= Xtil
            Yhat = Initial_2D
            Yhat[:,] -= Ytil

            U, s, V = svd(np.dot(Xhat.transpose(),Yhat))



            #M = np.dot(U,np.transpose(V))
            #projection[i] = (point - Xtil) * M + Ytil
        print(U.shape,s.shape,V.shape)
        return projection


    except Exception as e:
        print(traceback.print_exc())


    """

    projection = np.zeros((matrix_nRow,2))
    for i in range(matrix_nRow):
        point = matrix_dataset[i, :]

        skip = False
        alphas = np.zeros((Init_2D_Row))

        for j in range(samplesSubset):
            dist = sum(2 ** (matrix_dataset[j, :] - point))
            if dist < 1e-6:
                projection[i,:] = Initial_2D[j, :]
                skip = True
                break
            alphas[j] = 1.0/dist

        if skip is True:
            continue

        c = samplesSubset * ControlPoints
        if c < samplesSubset:
            index = alphas[::-1]
            j = c
            for j in range(samplesSubset):
                alphas[index[j]] = 0

        alphas_sum = sum(alphas)
        alphas_sqrt = np.sqrt(alphas)

        Xtil = (alphas * Xs) / alphas_sum
        Ytil = (alphas * Initial_2D) / alphas_sum

        Xhat = samplesSubset
        Xhat[:,] -= Xtil
        Yhat = Initial_2D
        Initial_2D[]
        """



def main():
    try:
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        values = data[:, range(b - 1)]
        pos = data[:, b - 1]
        print("Executando LAMP... ")
        y = LAMP(values)
        forceScheme.plot(y, pos)
        #print(y)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()