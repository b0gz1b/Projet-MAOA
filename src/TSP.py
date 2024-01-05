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
        self.dimension = dimension
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
    
    