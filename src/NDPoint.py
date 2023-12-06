from typing import List
import numpy as np


class DPoint:
	"""
	Point data structure for any multi-objective problem.
	"""
	def __init__(self, value: np.ndarray) -> None:
		"""
		Constructor of the Point class.
		:param value: the value of the point
		"""
		self.value = value
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return str(self.value)
	
	def __eq__(self, __value: object) -> bool:
		"""
		Checks if the current point is equal to another point.
		:param __value: the other point
		:return: True if the current point is equal to the other point, False otherwise
		"""
		return np.all(self.value == __value.value)

	def dominates(self, other: 'DPoint') -> bool:
		"""
		Checks if the current point dominates another point.
		:param other: the other point
		:return: True if the current point dominates the other point, False otherwise
		"""
		return np.all(self.value >= other.value) and np.any(self.value > other.value)
	
	def covers(self, other: 'DPoint') -> bool:
		"""
		Checks if the current point covers another point.
		:param other: the other point
		:return: True if the current point covers the other point, False otherwise
		"""
		return self.dominates(other) or np.all(self.value == other.value)
	
	def euclidean_distance(self, other: 'DPoint') -> float:
		"""
		Computes the euclidean distance between the current point and another point.
		:param other: the other point
		:return: the euclidean distance between the current point and another point
		"""
		return np.linalg.norm(self.value - other.value)
	
	def average_euclidean_distance(self, others: List['DPoint']) -> float:
		"""
		Computes the average euclidean distance between the current point and a list of other points.
		:param others: the list of other points
		:return: the average euclidean distance between the current point and a list of other points
		"""
		return np.mean([self.euclidean_distance(other) for other in others])