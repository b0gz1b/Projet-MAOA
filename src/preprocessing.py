import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans
from TSP import TSP
import matplotlib.pyplot as plt

def affinity_propagation(points, damping=0.5, max_iter=1000, convergence_iter=15, copy=True, preference=None,
                         affinity='euclidean', verbose=False, random_state=None):
    """
    Clusters the given points using the affinity propagation algorithm.
    :param points: the points to cluster
    :param damping: the damping factor
    :param max_iter: the maximum number of iterations
    :param convergence_iter: the number of iterations with no change in the number of clusters that stops the algorithm
    :param copy: if True, a copy of the points is used
    :param preference: the preference for each point
    :param affinity: the affinity metric to use
    :param verbose: if True, the algorithm will print information about the clustering process
    :return: the cluster centers and the labels of the points
    """
    af = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy,
                             preference=preference, affinity=affinity, verbose=verbose, random_state=random_state)
    af.fit(points)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return cluster_centers_indices, labels

def k_means(points, n_clusters=8, init='k-means++', n_init=50, max_iter=10000, tol=0.0001,
            verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    """
    Clusters the given points using the k-means algorithm.
    :param points: the points to cluster
    :param n_clusters: the number of clusters
    :param init: the initialization method
    :param n_init: the number of initializations to perform
    :param max_iter: the maximum number of iterations
    :param tol: the tolerance
    :param verbose: if True, the algorithm will print information about the clustering process
    :param random_state: the random state
    :param copy_x: if True, a copy of the points is used
    :param algorithm: the algorithm to use
    :return: the cluster centers and the labels of the points
    """
    km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
    km.fit(points)
    cluster_centers = km.cluster_centers_
    labels = km.labels_
    return cluster_centers, labels

def plot_km(points, labels, cluster_centers):
    """
    Plots the clustering obtained by the k-means algorithm.
    :param points: the points
    :param labels: the labels of the points
    :param cluster_centers: the cluster centers
    """
    fig, ax = plt.subplots()
    # K-means plot
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    # plot lines between points and their cluster centers
    for i in range(len(points)):
        ax.plot([points[i, 0], cluster_centers[labels[i], 0]], [points[i, 1], cluster_centers[labels[i], 1]], color='black', alpha=0.5, linewidth=0.5, linestyle='--')
    ax.set_title('K-means')
    plt.show()
    

if __name__ == '__main__':
    inst = TSP.from_file("TSP/Instances_TSP/eil76.tsp")

    cluster_centers_indices_af, labels_af = affinity_propagation(inst.points, damping=0.5)
    cluster_centers_km, labels_km = k_means(inst.points, len(cluster_centers_indices_af))
    rearrange = np.random.permutation(len(cluster_centers_indices_af))
    cluster_centers_indices_af = cluster_centers_indices_af[rearrange]
    labels_af = np.array([np.where(rearrange == i)[0][0] for i in labels_af])
    # Plot both methods on side by side plots
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Clustering')

    # K-means plot
    axs[0].scatter(inst["COORDS"][:, 0], inst["COORDS"][:, 1], c=labels_km)
    # plot lines between points and their cluster centers
    for i in range(len(inst["COORDS"])):
        axs[0].plot([inst["COORDS"][i, 0], cluster_centers_km[labels_km[i], 0]], [inst["COORDS"][i, 1], cluster_centers_km[labels_km[i], 1]], color='black', alpha=0.5, linewidth=0.5, linestyle='--')
    axs[0].set_title('K-means')

    # Affinity Propagation plot
    axs[1].scatter(inst["COORDS"][:, 0], inst["COORDS"][:, 1], c=labels_af)
    # plot lines between points and their cluster centers
    for i in range(len(inst["COORDS"])):
        axs[1].plot([inst["COORDS"][i, 0], inst["COORDS"][cluster_centers_indices_af[labels_af[i]], 0]], [inst["COORDS"][i, 1], inst["COORDS"][cluster_centers_indices_af[labels_af[i]], 1]], color='black', alpha=0.5, linewidth=0.5, linestyle='--')
    axs[1].set_title('Affinity propagation')
    plt.show()

    print("K-means cluster centers:\n", cluster_centers_km)
    print("Affinity propagation cluster centers:\n", inst["COORDS"][cluster_centers_indices_af])