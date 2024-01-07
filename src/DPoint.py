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
		self.value = np.asarray(value)
		self.dimension = len(value)
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return str(self.value)
	
	def __sub__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Subtracts two points.
		:param __value: the other point
		:return: the difference between the two points
		"""
		return DPoint(self.value - __value.value)
	
	def __add__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Adds two points.
		:param __value: the other point
		:return: the sum of the two points
		"""
		return DPoint(self.value + __value.value)
	
	def __mul__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Multiplies two points.
		:param __value: the other point
		:return: the product of the two points
		"""
		return DPoint(self.value * __value.value)
	
	def __eq__(self, __value: 'DPoint') -> bool:
		"""
		Checks if the current point is equal to another point.
		:param __value: the other point
		:return: True if the current point is equal to the other point, False otherwise
		"""
		return np.all(self.value == __value.value)

	def __len__(self) -> int:
		"""
		Gives the length of the point.
		:return: the length of the point
		"""
		return len(self.value)
	
	def __hash__(self) -> int:
		"""
		Computes the hash of the point.
		:return: the hash of the point
		"""
		return hash(tuple(self.value))

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
	
	def average_euclidean_distance(self, others: list['DPoint']) -> float:
		"""
		Computes the average euclidean distance between the current point and a list of other points.
		:param others: the list of other points
		:return: the average euclidean distance between the current point and a list of other points
		"""
		return np.mean([self.euclidean_distance(other) for other in others])
	
	def sum(self) -> float:
		"""
		Computes the sum of the values of the point.
		:return: the sum of the values of the point
		"""
		return np.sum(self.value)
	
	def weighted_sum(self, weights: np.ndarray) -> float:
		"""
		Computes the weighted sum of the values of the point.
		:param weights: the weights
		:return: the weighted sum of the values of the point
		"""
		if len(weights) != len(self.value):
			raise ValueError("The length of the weights vector should be equal to the length of the value vector.")
		return np.dot(self.value, weights)
	
	def owa(self, weights: np.ndarray, direction: str = "max") -> float:
		"""
		Computes the ordered weighted average of the values of the point.
		:param weights: the weights
		:param direction: the direction of the OWA, either "max" or "min", default is "max"
		:return: the ordered weighted average of the values of the point
		"""
		if len(weights) != len(self.value):
			raise ValueError("The length of the weights vector should be equal to the length of the value vector.")
		if direction == "max":
			return np.dot(np.sort(self.value), weights)
		elif direction == "min":
			return np.dot(np.flip(np.sort(self.value)), weights)
	
	def choquet(self, cap, direction: str = "max") -> float:
		"""
		Computes the Choquet integral of the values of the point.
		:param cap: the capacity
		:param direction: the direction of the Choquet integral, either "max" or "min", default is "max"
		:return: the Choquet integral of the values of the point
		"""
		if direction == "max":
			x_p = np.sort(self.value)
			arg_x_p = np.argsort(self.value)
		elif direction == "min":
			x_p = np.flip(np.sort(self.value))
			arg_x_p = np.flip(np.argsort(self.value))
		cv = x_p[0]
		for i in range(1, len(self.value)):
			X = set(arg_x_p[i:])
			cv += (x_p[i] - x_p[i-1]) * cap.v(X)
		return cv
	
	def evaluate(self, dm, pref_model: str) -> float:
		"""
		Computes the value of the point according to a preference model.
		:param dm: the decision maker, either the weights or the capacity
		:param pref_model: the preference model, either "ws", "owa" or "choquet"
		:return: the value of the point according to the preference model
		"""
		if pref_model == "ws":
			return self.weighted_sum(dm)
		elif pref_model == "owa":
			return self.owa(dm)
		elif pref_model == "choquet":
			return self.choquet(dm)
		else:
			raise ValueError("Unknown preference model.")
				
		
if __name__ == "__main__":
	x = DPoint([10, 5, 15])
	y = DPoint([10, 12, 8])
	d1 = {" ": 0, "0": 0.2, "1": 0.1, "2": 0.3, "0,1": 0.45, "0,2": 0.5, "1,2": 0.65, "0,1,2": 1}
	c1 = lambda s : d1[",".join(map(str,np.sort(list(s))))]
	d2 = {" ": 0, "0": 0.35, "1": 0.5, "2": 0.55, "0,1": 0.7, "0,2": 0.9, "1,2": 0.8, "0,1,2": 1}
	c2 = lambda s : d2[",".join(map(str,np.sort(list(s))))]
	print(x.choquet(c1))
	print(y.choquet(c1))
	print(x.choquet(c2))
	print(y.choquet(c2))