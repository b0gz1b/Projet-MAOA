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
        self.d = 3
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
        return cls(tsp.name, tsp.d, tsp.points, clusters)
    
    @classmethod
    def from_file(cls, filename: str) -> 'GTSP':
        """
        Constructor of the GTSP class.
        :param filename: the name of the file containing the GTSP instance
        :return: a GTSP instance
        """
        tsp = TSP.from_file(filename)
        cluster_centers_indices_af, labels_af = pre.affinity_propagation(tsp.points, damping=0.5)
        _, labels_km = pre.k_means(tsp.points, len(cluster_centers_indices_af))
        clusters = [[] for _ in range(len(cluster_centers_indices_af))]
        for i in range(len(labels_km)):
            clusters[labels_km[i]].append(i)
        return cls(tsp.name, tsp.d, tsp.points, clusters)
    
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
    
    def cost_time_ratio(self, tour: list[int]) -> tuple[float, float, float]:
        """
        Computes the cost, the time and the ratio of a tour.
        :param tour: the tour
        :return: the cost, the time and the ratio of the tour
        """
        t = 0
        r = 0
        n = 0
        d = 0
        for i in range(len(tour)):
            d += np.linalg.norm(self.points[tour[i]] - self.points[tour[(i + 1) % len(tour)]])
        for i in range(len(self.points)):
            if i not in tour:
                d += self.find_closest_station(i, tour)[1] * 10
            for j in range(i+1, len(self.points)):
                _, ratio, dist = self.itinerary(i, j, tour)
                t += dist
                r += ratio
                n += 1
        return d, t/n, r/n

    def find_closest_station(self, i: int, tour: list[int]) -> tuple[int, float]:
        min_dist = np.inf
        min_point = None
        for k in range(len(tour)):
            dist = np.linalg.norm(self.points[i] - self.points[tour[k]])
            if dist < min_dist:
                min_dist = dist
                min_point = k
        return min_point, min_dist
  
    def intersections(self, tour: list[int]) -> int:
        """
        Computes the number of intersections in a tour.
        :param tour: the tour
        :return: the number of intersections in the tour
        """
        def onSegment(p, q, r): 
            px, py = p
            qx, qy = q
            rx, ry = r
            return (qx <= max(px, rx)) and (qx >= min(px, rx)) and (qy <= max(py, ry)) and (qy >= min(py, ry))
        
        def orientation(p, q, r): 
            px, py = p
            qx, qy = q
            rx, ry = r
            val = (float(qy - py) * (rx - qx)) - (float(qx - px) * (ry - qy)) 
            if (val > 0): 
                return 1
            elif (val < 0): 
                return 2
            else:
                return 0
        def doIntersect(p1,q1,p2,q2):
            o1 = orientation(p1, q1, p2) 
            o2 = orientation(p1, q1, q2) 
            o3 = orientation(p2, q2, p1) 
            o4 = orientation(p2, q2, q1) 
            if ((o1 != o2) and (o3 != o4)): 
                return True
            if ((o1 == 0) and onSegment(p1, p2, q1)): 
                return True
            if ((o2 == 0) and onSegment(p1, q2, q1)): 
                return True
            if ((o3 == 0) and onSegment(p2, p1, q2)): 
                return True
            if ((o4 == 0) and onSegment(p2, q1, q2)): 
                return True
            return False
        intersections = 0
        for i in range(len(tour)):
            for k in range(i+2, len(tour)):
                j = (i + 1) % len(tour)
                l = (k + 1) % len(tour)
                # check if the segments (i,j) and (k,l) intersect
                a, b, c, d = self.points[tour[i]], self.points[tour[j]], self.points[tour[k]], self.points[tour[l]]
                if doIntersect(a, b, c, d):
                    intersections += 1
        return intersections

    def itinerary(self, i: int, j: int, tour: list[int]) -> tuple[list[int], float, float]:
        station_1, dist_station_1 = self.find_closest_station(i, tour)
        station_2, dist_station_2 = self.find_closest_station(j, tour)
        pied = 10 * np.linalg.norm(self.points[i] - self.points[j])
        # compute the to ways to go from station_1 to station_2 with the tour
        base = 10 * (dist_station_1 + dist_station_2)
        dist_metro_p = base
        dist_metro_m = base
        ratio_metro_p = 1
        ratio_metro_m = 1
        prev_station_p = [station_1]
        prev_station_m = [station_1]
        not_arrived_p = True
        not_arrived_m = True
        for k in range(len(tour)):
            cur_station_p = (station_1 + k) % len(tour)
            cur_station_m = (station_1 - k) % len(tour)
            if not_arrived_p:
                d = np.linalg.norm(self.points[tour[prev_station_p[-1]]] - self.points[tour[cur_station_p]])
                dist_metro_p += d
                ratio_metro_p += d
                if len(prev_station_p) > 1:
                    prev_station_p.append(cur_station_p)
                if cur_station_p == station_2:
                    not_arrived_p = False
            if not_arrived_m:
                d = np.linalg.norm(self.points[tour[prev_station_m[-1]]] - self.points[tour[cur_station_m]])
                dist_metro_m += d
                ratio_metro_m += d
                if len(prev_station_m) > 1:
                    prev_station_m.append(cur_station_m)
                if cur_station_m == station_2:
                    not_arrived_m = False
        if min(dist_metro_p, dist_metro_m) >= pied:
            return [], 0, pied
        elif dist_metro_p < dist_metro_m:
            return prev_station_p, base / ratio_metro_p, dist_metro_p
        else:
            return prev_station_m, base / ratio_metro_m, dist_metro_m


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