import numpy as np
from itertools import combinations

class Capacity:
    def __init__(self, d: int, dict_capacity: dict) -> None:
        """
        Constructor of the Capacity class.
        :param d: the dimension of the problem
        :param dict_capacity: the dictionary of capacities
        """
        self.d = d
        self.dict_capacity = dict_capacity
        self.moebius_inverse = {}
        for i in range(self.d + 1):
            for A in combinations(range(self.d), i):
                key = ",".join(map(str, A))
                self.moebius_inverse[key] = 0
                if i != 0:
                    for j in range(i+1):
                        for B in combinations(A, j):
                            self.moebius_inverse[key] += (-1)**(i - j) * self.dict_capacity[",".join(map(str, B))]

    @classmethod
    def from_moebius_inverse(cls, d: int, moebius_inverse: dict) -> 'Capacity':
        """
        Constructor of the Capacity class.
        :param d: the dimension of the problem
        :param moebius_inverse: the dictionary of Moebius inverses
        :return: a Capacity instance
        """
        dict_capacity = {"": 0, ",".join(map(str, range(d))): 1}
        for size_A in range(1, d):
            for A in combinations(range(d), size_A):
                key = ",".join(map(str, A))
                dict_capacity[key] = 0
                for size_B in range(size_A + 1):
                    for B in combinations(A, size_B):
                        key_moebius = ",".join(map(str, B))
                        dict_capacity[key] += moebius_inverse[key_moebius]
        return cls(d, dict_capacity)

    def v(self, S: set) -> float:
        """
        Computes the capacity of a set.
        :param S: the set
        :return: the capacity of the set
        """
        return self.dict_capacity[",".join(map(str,np.sort(list(S))))]
    
    def m(self, S: set) -> float:
        """
        Computes the Moebius inverse of a set.
        :param S: the set
        :return: the Moebius inverse of the set
        """
        return self.moebius_inverse[",".join(map(str,np.sort(list(S))))]
