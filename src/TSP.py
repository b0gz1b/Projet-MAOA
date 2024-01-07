from matplotlib import pyplot as plt
import numpy as np

class TSP:
    """
    TSP data structure.
    """
    def __init__(self, name: str, dimension: int, points: np.ndarray) -> None:
        """
        Constructor of the TSP class.
        :param name: the name of the instance
        :param dimension: the dimension of the instance
        :param points: the points of the instance
        :param distance_matrix: the distance matrix of the instance
        """
        self.name = name
        self.d = dimension
        self.points = points

    @classmethod
    def from_file(cls, filename: str) -> 'TSP':
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("NAME"):
                    name = line.split(":")[1].strip()
                elif line.startswith("DIMENSION"):
                    dimension = int(line.split(":")[1].strip())
                    points = np.zeros((dimension, 2))
                elif line == "NODE_COORD_SECTION":
                    for coord_line in file:
                        coord_line = coord_line.strip()
                        if coord_line == "EOF" or coord_line == "":
                            break
                        parts = coord_line.split()
                        node_id = int(parts[0])
                        x_coord = float(parts[1])
                        y_coord = float(parts[2])
                        points[node_id - 1] = [x_coord, y_coord]
        return cls(name, dimension, points)
    
    def plot_instance(self, file_path=None):
        """
        Plots the instance.
        """
        fig, ax = plt.subplots()
        ax.scatter(self.points[:, 0], self.points[:, 1])
        if file_path is not None:
            plt.savefig(file_path)
        else:
            plt.show()

    def plot_tour(self, tour, file_path=None):
        """
        Plots a tour.
        :param tour: the tour
        """
        fig, ax = plt.subplots()
        ax.scatter(self.points[:, 0], self.points[:, 1])
        for i in range(len(tour)):
            ax.plot([self.points[tour[i]][0], self.points[tour[(i + 1) % len(tour)]][0]], [self.points[tour[i]][1], self.points[tour[(i + 1) % len(tour)]][1]], color='red')
        if file_path is not None:
            plt.savefig(file_path)
        else:
            plt.show()

if __name__ == "__main__":
    inst = TSP.from_file("TSP/Instances_TSP/att48.tsp")
    inst.plot_instance("tmp/att48.png")
    inst = TSP.from_file("TSP/Instances_TSP/eil76.tsp")
    inst.plot_instance("tmp/eil76.png")
    inst = TSP.from_file("TSP/Instances_TSP/rd100.tsp")
    inst.plot_instance("tmp/rd100.png")

    
    