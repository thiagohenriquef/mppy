def neighborhood_preservation(data_matrix, data_proj, clusters, max_neighbors=30):
    from scipy.spatial.distance import squareform, pdist
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    instances = data_matrix.shape[0]
    dist_matrix = squareform(pdist(data_matrix))
    dist_proj = squareform(pdist(data_proj))
    idx_matrix = np.argsort(dist_matrix)[:, 1:max_neighbors + 1]
    idx_proj = np.argsort(dist_proj)[:, 1:max_neighbors + 1]
    values = np.zeros((max_neighbors))

    for n in range(max_neighbors):
        percentage = 0.0

        for i in range(instances):
            total = 0.0
            for j in range(n):
                if idx_matrix[i,j] in idx_proj:
                    total = total + 1

            percentage += total / (n + 1)
        values[n] = percentage / instances

    return values

def neighborhood_hit(data_proj, clusters, max_neighbors=15):
    from scipy.spatial.distance import squareform, pdist
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    instances = data_proj.shape[0]
    dist_proj = squareform(pdist(data_proj))
    neighbors = np.argsort(dist_proj)[:, 1:max_neighbors + 1]
    values = np.zeros((max_neighbors))

    for n in range(max_neighbors):
        percentage = 0.0
        for i in range(instances):
            c = clusters[i]
            total = 0.0
            for j in range(n+1):
                point = neighbors[i,j]
                if c == clusters[point]:
                    total = total + 1

            percentage += total / (n + 1)

        values[n] = percentage / instances
    return values
