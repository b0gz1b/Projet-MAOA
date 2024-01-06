from DPoint import DPoint
from GTSP import GTSP
import numpy as np
import multiprocessing as mp

class DGTSPPoint(DPoint):
	def __init__(self, gtsp: 'GTSP', tour: list[int] = [], cluster_tour: list[int] = []) -> None:
		"""
		Constructor of the Point class.
		:param gtsp: the GTSP instance
		:param tour: the tour corresponding to the point
		"""
		super().__init__(-np.asarray(gtsp.cost_time_ratio(tour)))
		self.gtsp = gtsp
		self.tour = tour
		self.cluster_tour = cluster_tour

	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return np.array2string(self.value)
	
	def neighbors_2opt(self) -> list['DGTSPPoint']:
		neigh = []
		cpu_count = mp.cpu_count()
		with mp.Pool(cpu_count) as p:
			res = p.starmap(aux_n2opt, [(self, c_alpha_ind, c_gamma_ind) for c_alpha_ind in range(len(self.cluster_tour) - 2) for c_gamma_ind in range(c_alpha_ind + 2, len(self.cluster_tour))])
		for r in res:
			neigh.append(DGTSPPoint(self.gtsp, r[0], r[1]))
		return neigh
	
def aux_n2opt(point, c_alpha_ind, c_gamma_ind):
	c_beta_ind = (c_alpha_ind + 1) % len(point.cluster_tour)
	c_delta_ind = (c_gamma_ind + 1) % len(point.cluster_tour)
	prec_alpha = point.tour[(c_alpha_ind - 1) % len(point.tour)]
	next_beta = point.tour[(c_beta_ind + 1) % len(point.tour)]
	prec_gamma = point.tour[(c_gamma_ind - 1) % len(point.tour)]
	next_delta = point.tour[(c_delta_ind + 1) % len(point.tour)]
	u,v,w,z = None, None, None, None
	min_d_uw = np.inf
	min_d_vz = np.inf
	for a in point.gtsp.clusters[point.cluster_tour[c_alpha_ind]]:
		for b in point.gtsp.clusters[point.cluster_tour[c_gamma_ind]]:
			d_ia = np.linalg.norm(point.gtsp.points[prec_alpha] - point.gtsp.points[a])
			d_ab = np.linalg.norm(point.gtsp.points[a] - point.gtsp.points[b])
			d_bh = np.linalg.norm(point.gtsp.points[b] - point.gtsp.points[prec_gamma])
			if d_ia + d_ab + d_bh < min_d_uw:
				min_d_uw = d_ia + d_ab + d_bh
				u = a
				w = b
	for a in point.gtsp.clusters[point.cluster_tour[c_beta_ind]]:
		for b in point.gtsp.clusters[point.cluster_tour[c_delta_ind]]:
			d_ja = np.linalg.norm(point.gtsp.points[next_beta] - point.gtsp.points[a])
			d_ab = np.linalg.norm(point.gtsp.points[a] - point.gtsp.points[b])
			d_bk = np.linalg.norm(point.gtsp.points[b] - point.gtsp.points[next_delta])
			if d_ja + d_ab + d_bk < min_d_vz:
				min_d_vz = d_ja + d_ab + d_bk
				v = a
				z = b
	new_tour = point.tour.copy()
	new_tour[c_alpha_ind] = u
	new_tour[c_beta_ind] = w
	new_tour[c_gamma_ind] = v
	new_tour[c_delta_ind] = z
	new_cluster_tour = point.cluster_tour.copy()
	new_cluster_tour[c_beta_ind] = point.cluster_tour[c_gamma_ind]
	new_cluster_tour[c_gamma_ind] = point.cluster_tour[c_beta_ind]
	return new_tour, new_cluster_tour