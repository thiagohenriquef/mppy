def neighborhood_preservation(data_matrix, data_proj, max_neighbors=30):
    from scipy.spatial.distance import squareform, pdist
    import numpy as np
    
    instances = data_matrix.shape[0]
    dist_matrix = squareform(pdist(data_matrix))
    dist_proj = squareform(pdist(data_proj))
    idx_matrix = dist_matrix.argsort()[:, 1:max_neighbors+1]
    idx_proj = dist_proj.argsort()[:, 1:max_neighbors+1]

    values = np.zeros((max_neighbors))
    
    for n in range(0,max_neighbors):
        percentage = 0.0

        for i in range(instances):
            total = 0.0
            aux_matrix = idx_matrix[i,:]
            aux_proj = idx_proj[i,:]
            
            for j in range(n):
                if aux_proj[j] in aux_matrix:
                    total = total + 1
                
            percentage += total / (n+1)

        values[n] = percentage / instances

    import matplotlib.pyplot as plt
    plt.title("Neighborhood Preservation")
    plt.xlabel('Neighbors')
    plt.ylabel('Percentage')
    ax = plt.subplot(111)
    ax.set_xlim(0,1)
    dim = np.arange(0, values.shape[0], 1)
    plt.plot(values, linestyle='-', marker='^', color='b')
    plt.ylim(ymin=0, ymax=1)
    plt.xticks(dim)
    plt.grid()
    plt.show()
    plt.close()

def neighborhood_hit(data_proj, clusters, max_neighbors=30):
    from scipy.spatial.distance import squareform, pdist
    import numpy as np
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    
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
