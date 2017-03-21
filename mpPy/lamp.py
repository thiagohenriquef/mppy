import numpy as np
import forceScheme
import sys
import traceback
from scipy.linalg import svd
from scipy.spatial.distance import squareform, pdist

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
            aux = np.zeros((matrix_nCol,Init_2D_Col))
            for k in range(Init_2D_Col):
                aux[k, range(Init_2D_Col)] = V[k]

            M = np.dot(U,aux)
            projection[i] = np.dot((point - Xtil), M)  + Ytil
        return projection


    except Exception as e:
        print(traceback.print_exc())

def main():
    try:
        print("Executando LAMP... ")
        #file = str(sys.argv[1])
        #data = rf.readInput(file)
        data = readInput("iris.data")
        a, b = data.shape
        new_matrix = data[:, range(b - 1)]
        positions = data[:, b - 1]

        random_matrix = np.random.random((a, 2))
        test = random_matrix.copy()
        new_matrix = LAMP(new_matrix)

        stress = kruskal(test, new_matrix)

        plot(new_matrix, positions)
        print(stress)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()