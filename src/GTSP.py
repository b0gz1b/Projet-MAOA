from matplotlib import pyplot as plt
import numpy as np
import preprocessing as pre
from TSP import TSP

class GTSP:
    """
    A class representing a GTSP instance.
    """
    def __init__(self, name: str, dimension: int, points: np.ndarray, clusters: list[list[int]]) -> None:
        """
        Constructor of the GTSP class.
        :param name: the name of the instance
        :param dimension: the dimension of the instance
        :param points: the points of the instance
        :param clusters: the clusters of the instance, represented as a list of lists of indices
        """
        self.name = name
        self.dimension = dimension
        self.points = points
        self.clusters = clusters
    
    @classmethod
    def from_TSP(cls, tsp: TSP) -> 'GTSP':
        """
        Constructor of the GTSP class.
        :param tsp: the TSP instance
        :return: a GTSP instance
        """
        cluster_centers_indices_af, _ = pre.affinity_propagation(tsp.points, damping=0.5)
        _, labels_km = pre.k_means(tsp.points, len(cluster_centers_indices_af))
        clusters = [[] for _ in range(len(cluster_centers_indices_af))]
        for i in range(len(labels_km)):
            clusters[labels_km[i]].append(i)
        return cls(tsp.name, tsp.dimension, tsp.points, clusters)
    
    @classmethod
    def from_file(cls, filename: str) -> 'GTSP':
        """
        Constructor of the GTSP class.
        :param filename: the name of the file containing the GTSP instance
        :return: a GTSP instance
        """
        tsp = TSP.from_file(filename)
        cluster_centers_indices_af, _ = pre.affinity_propagation(tsp.points, damping=0.5)
        _, labels_km = pre.k_means(tsp.points, len(cluster_centers_indices_af))
        clusters = [[] for _ in range(len(cluster_centers_indices_af))]
        for i in range(len(labels_km)):
            clusters[labels_km[i]].append(i)
        return cls(tsp.name, tsp.dimension, tsp.points, clusters)
    
    def cluster_dist(self, i: int, j: int) -> tuple[float, tuple[int, int]]:
        """
        Computes the distance between two clusters.
        :param i: the first cluster
        :param j: the second cluster
        :return: the distance between the two clusters
        """
        min_dist = np.inf
        min_points = None
        for k in self.clusters[i]:
            for l in self.clusters[j]:
                dist = np.linalg.norm(self.points[k] - self.points[l])
                if dist < min_dist:
                    min_dist = dist
                    min_points = (k, l)
        return min_dist, min_points
    
    def plot_tour(self, tour, file_path=None):
        """
        Plots a tour.
        :param tour: the tour
        """
        fig, ax = plt.subplots()
        # plot the points in each cluster except the ones in the tour
        for k in range(len(self.clusters)):
            x_s = []
            y_s = []
            for i in range(len(self.clusters[k])):
                if self.clusters[k][i] not in tour:
                    x_s.append(self.points[self.clusters[k][i], 0])
                    y_s.append(self.points[self.clusters[k][i], 1])
            ax.scatter([x_s], [y_s])
        # plot the points in the tour as triangles
        x_s = [None] * len(tour)
        y_s = [None] * len(tour)
        plt.gca().set_prop_cycle(None)
        for i in range(len(tour)):
            # find the cluster of the point
            for k in range(len(self.clusters)):
                if tour[i] in self.clusters[k]:
                    x_s[k] = self.points[tour[i], 0]
                    y_s[k] = self.points[tour[i], 1]
        for k in range(len(self.clusters)):
            ax.scatter([x_s[k]], [y_s[k]], marker='^')
        # plot the tour edges
        for i in range(len(tour)):
            ax.plot([self.points[tour[i], 0], self.points[tour[(i + 1) % len(tour)], 0]], [self.points[tour[i], 1], self.points[tour[(i + 1) % len(tour)], 1]], color='black')
        # plot fine lines between the points and the closest points in the tour
        for i in range(len(self.points)):
            if i not in tour:
                min_dist = np.inf
                min_point = None
                for j in range(len(tour)):
                    dist = np.linalg.norm(self.points[i] - self.points[tour[j]])
                    if dist < min_dist:
                        min_dist = dist
                        min_point = tour[j]
                ax.plot([self.points[i, 0], self.points[min_point, 0]], [self.points[i, 1], self.points[min_point, 1]], color='red', alpha=0.3, linewidth=0.5, linestyle='-.')
        
        ax.set_title('Tour')
        if file_path is None:
            plt.savefig("tmp/tour.png")
        else:    
            plt.savefig(file_path)