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
