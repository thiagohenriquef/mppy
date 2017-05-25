def neighborhood_preservation(data_matrix, data_proj, max_neighbors=50):
    from scipy.spatial.distance import squareform, pdist
    import numpy as np
    
    values = np.zeros((max_neighbors))
    instances = data_matrix.shape[0]
    
    dist_matrix = squareform(pdist(data_matrix))
    dist_proj = squareform(pdist(data_proj))
    
    n_proj = dist_proj.argsort()[: ,:(max_neighbors+1)]
    n_data = dist_matrix.argsort()[: , :(max_neighbors+1)]
    
    for n in range(max_neighbors):
        percentage = 0.0

        for i in range(instances):
            total = 0.0
            aux_n_proj = n_proj[i,:]
            aux_n_data = n_data[i,:]
            
            for j in range(n+1):
                if aux_n_proj[j] in aux_n_data:
                    total = total + 1

            percentage += total / (n+1)

        values[n] = percentage / instances

    values = 1 - values
    import matplotlib.pyplot as plt
    plt.title("Neighborhood Preservation")
    plt.xlabel('Neighbors')
    plt.ylabel('Percentage')
    ax = plt.subplot(111)
    #ax.set_xlim(0,1)
    dim = np.arange(0, values.shape[0], 1)
    plt.plot(values, linestyle='-', marker='^', color='b')
    #plt.ylim(ymin=0, ymax=1)
    plt.xticks(dim)
    plt.grid()
    plt.show()
    plt.close()
    #return values

def neighborhood_hit(data_proj, clusters, max_neighbors=30):
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
                    total += 1

            percentage += total / (n + 1)

        values[n] = percentage / instances

    import matplotlib.pyplot as plt
    plt.title("Neighborhood Hit")
    plt.xlabel('Neighbors')
    plt.ylabel('Percentage')
    ax = plt.subplot(111)
    #ax.set_xlim(0,1)
    dim = np.arange(0, values.shape[0], 1)
    plt.plot(values, linestyle='--', marker='^', color='b')
    #plt.ylim(ymin=0, ymax=1)
    plt.xticks(dim)
    plt.grid()
    plt.show()
    plt.close()
    #return values
