import time
import numpy as np
from typing import List
from GTSP import GTSP
from DGTSPPoint import DGTSPPoint
import heuristique as heur
from NDTree import NDTree

NUMBER_OF_CHILDREN = lambda d: d + 1
MAX_LEAF_SIZE = 20

def P0(gtsp: GTSP, verbose: int = 0) -> NDTree:
	"""
	Performs the P0 algorithm on the instance of the problem
	:param gtsp: the instance of the problem
	:param verbose: the verbosity level, 0 for no verbosity, 1 for local verbosity, 2 for global verbosity
	:return: the initial population
	"""
	p0 = NDTree(gtsp.d, NUMBER_OF_CHILDREN(gtsp.d), MAX_LEAF_SIZE)
	p0.update(heur.farthest_insertion(gtsp), verbose = True if verbose == 2 else False)
	p0.update(heur.nearest_insertion(gtsp), verbose = True if verbose == 2 else False)
	for i in range(20):
		p0.update(heur.random_tour(gtsp), verbose = True if verbose == 2 else False)
	return p0

def PLS(gtsp: GTSP, initial_pop: NDTree = None, verbose: int = 0) -> List[DGTSPPoint]:
	"""
	Performs the PLS algorithm on the instance of the problem
	:param gtsp: the instance of the problem
	:param initial_pop: the initial population
	:param verbose: the verbosity level, 0 for no verbosity, 1 for local verbosity, 2 for global verbosity
	:return: the efficient set approximation
	"""
	start = time.time()
	if initial_pop is None:
		initial_pop = P0(gtsp) # Initial population
	efficient_set_approx = initial_pop.copy() # Pareto front archive
	current_pop = initial_pop.copy() # Current population
	aux_pop = NDTree(gtsp.d, NUMBER_OF_CHILDREN(gtsp.d), MAX_LEAF_SIZE) # Auxiliary population
	while current_pop.get_pareto_front() != []:
		size_of_p = len(current_pop.get_pareto_front())
		start_it = time.time()
		_current_pop = current_pop.get_pareto_front()
		for i, p in enumerate(_current_pop):
			start_p = time.time()
			neighbors = p.neighbors_2opt()
			for j, neighbor in enumerate(neighbors):
				if not p.covers(neighbor):
					if efficient_set_approx.update(neighbor, verbose = True if verbose == 2 else False):
						aux_pop.update(neighbor, verbose = True if verbose == 2 else False)
			end_p = time.time()
			if verbose != 0:
				print("Iteration {}/{}: {} neighbors generated and explored in {:.2f}s".format(i+1, len(_current_pop), len(neighbors), end_p-start_p))
		current_pop = aux_pop.copy()
		aux_pop = NDTree(gtsp.d, NUMBER_OF_CHILDREN(gtsp.d), MAX_LEAF_SIZE)
		print("Size of P: {}, iteration time: {:.2f}s, total time: {:.2f}s".format(size_of_p, time.time()-start_it, time.time()-start))
	return efficient_set_approx.get_pareto_front()

def select(gtsp: GTSP, population: List[DGTSPPoint]) -> DGTSPPoint:
	"""
	Selects the least intersecting point from the population
	:param gtsp: the instance of the problem
	:param population: the population
	:return: the selected point
	"""
	min_intersections = np.inf
	best_point = None
	for point in population:
		intersections = gtsp.intersections(point.tour)
		if intersections < min_intersections:
			min_intersections = intersections
			best_point = point
	return best_point
	
if __name__ == "__main__":
	name = "st70"
	time_start = time.time()
	inst = GTSP.from_file(f"TSP/Instances_TSP/{name}.tsp")
	fi = heur.farthest_insertion(inst)
	ni = heur.nearest_insertion(inst)
	print(fi)
	print(ni)
	sols = PLS(inst)
	print(len(sols))
	print(sols)

	selected = select(inst, sols)
	print(inst.cost_time_ratio(selected.tour))
	# inst.plot_tour(selected.tour, f"tmp/selected_{name}.png")
	print(f"Time: {time.time()-time_start}")
	i = 0
	for sol in sols:
		inst.plot_tour(sol.tour, f"tmp/tour_{name}_{i}.png")
		print(inst.cost_time_ratio(sol.tour))
		i += 1

