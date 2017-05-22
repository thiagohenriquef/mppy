def neighborhood_preservation(data_matrix, data_proj, clusters, max_neighbors=15):
    from scipy.spatial.distance import squareform, pdist
    import numpy as np

    instances = data_matrix.shape[0]
    dist_matrix = squareform(pdist(data_matrix))
    dist_proj = squareform(pdist(data_proj))
    idx_matrix = dist_matrix.argsort()[:, 1:max_neighbors+1]
    idx_proj = dist_proj.argsort()[:, 1:max_neighbors+1]
    values = np.zeros((max_neighbors))

    for i in range(max_neighbors):
        percentage = 0.0

        for j in range(instances):
            total = 0.0
            aux_matrix = idx_matrix[i,:]
            aux_proj = idx_proj[i,:]

            for k in range(i):
                if aux_matrix[k] in aux_proj:
                    total = total + 1

            percentage += total / (i + 1)

        values[i] = percentage / instances

    import matplotlib.pyplot as plt
    plt.plot(values, linestyle='--', marker='o', color='b')
    #return values

def neighborhood_hit(data_proj, clusters, max_neighbors=15):
    from scipy.spatial.distance import squareform, pdist
    import numpy as np

    instances = data_proj.shape[0]
    dist_proj = squareform(pdist(data_proj))
    neighbors = dist_proj.argsort()[:,1:max_neighbors+1]
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
    import matplotlib.pyplot as plt
    plt.plot(values, linestyle='--', marker='o', color='b')
    return values
