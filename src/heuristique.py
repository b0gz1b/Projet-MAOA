from time import time
from GTSP import GTSP
from DGTSPPoint import DGTSPPoint, aux_n2opt
import numpy as np
import heapq as hq

def random_tour(gtsp: GTSP) -> DGTSPPoint:
    """
    Computes a random tour.
    :param gtsp: the GTSP instance
    :return: the random tour and the corresponding cluster tour
    """
    tour = []
    cluster_tour = []
    for i in range(len(gtsp.clusters)):
        tour.append(np.random.choice(gtsp.clusters[i]))
    np.random.shuffle(tour)
    for i in range(len(tour)):
        for j in range(len(gtsp.clusters)):
            if tour[i] in gtsp.clusters[j]:
                cluster_tour.append(j)
    return DGTSPPoint(gtsp, tour, cluster_tour)

def farthest_insertion(gtsp: GTSP) -> DGTSPPoint:
    """
    Computes a farthest insertion tour.
    :param gtsp: the GTSP instance
    :return: the farthest insertion tour and the corresponding cluster tour
    """
    cluster_dists = []
    # Compute the distance between each pair of clusters and store them in a heap
    for i in range(len(gtsp.clusters)-1):
        for j in range(i+1,len(gtsp.clusters)):
            d, p = gtsp.cluster_dist(i, j)
            hq.heappush(cluster_dists, (-d, (i, j), p))
    # Initialize the tour with the first pair of clusters in the heap
    _, (i, j), (p1, p2) = hq.heappop(cluster_dists)
    reserve = []
    tour = [p1, p2]
    cluster_tour = [i, j]
    # While there are still clusters to visit
    while len(cluster_tour) < len(gtsp.clusters):
        next_pair = hq.heappop(cluster_dists)
        _, (i, j), (p1, p2) = next_pair
        # If both clusters of the pair have already been visited, ignore it
        if i in cluster_tour and j in cluster_tour:
            continue
        # If only one of the clusters has already been visited, add the other one to the tour
        elif i in cluster_tour:
            tour.insert(0, p2)
            cluster_tour.insert(0, j)
            while len(reserve) > 0:
                hq.heappush(cluster_dists, reserve.pop()) 
        elif j in cluster_tour:
            tour.append(p1)
            cluster_tour.append(i)
            while len(reserve) > 0:
                hq.heappush(cluster_dists, reserve.pop())
        # If none of the clusters have already been visited, add the pair to the reserve
        else:
            reserve.append(next_pair)
    return DGTSPPoint(gtsp, tour, cluster_tour)
def nearest_insertion(gtsp: GTSP) -> DGTSPPoint:
    """
    Computes a nearest insertion tour.
    :param gtsp: the GTSP instance
    :return: the nearest insertion tour and the corresponding cluster tour
    """
    cluster_dists = []
    # Compute the distance between each pair of clusters and store them in a heap
    for i in range(len(gtsp.clusters)-1):
        for j in range(i+1,len(gtsp.clusters)):
            d, p = gtsp.cluster_dist(i, j)
            hq.heappush(cluster_dists, (d, (i, j), p))
    # Initialize the tour with the first pair of clusters in the heap
    _, (i, j), (p1, p2) = hq.heappop(cluster_dists)
    reserve = []
    tour = [p1, p2]
    cluster_tour = [i, j]
    # While there are still clusters to visit
    while len(cluster_tour) < len(gtsp.clusters):
        next_pair = hq.heappop(cluster_dists)
        _, (i, j), (p1, p2) = next_pair
        # If both clusters of the pair have already been visited, ignore it
        if i in cluster_tour and j in cluster_tour:
            continue
        # If only one of the clusters has already been visited, add the other one to the tour
        elif i in cluster_tour:
            tour.insert(0, p2)
            cluster_tour.insert(0, j)
            while len(reserve) > 0:
                hq.heappush(cluster_dists, reserve.pop()) 
        elif j in cluster_tour:
            tour.append(p1)
            cluster_tour.append(i)
            while len(reserve) > 0:
                hq.heappush(cluster_dists, reserve.pop())
        # If none of the clusters have already been visited, add the pair to the reserve
        else:
            reserve.append(next_pair)
    return DGTSPPoint(gtsp, tour, cluster_tour)
def RP1_procedure(gtsp: GTSP, tour: list[int], cluster_tour: list[int]) -> tuple[list[int], list[int]]:
    """
    Apply the RP1 procedure to refine a tour (2-opt generalization)
    :param gtsp: the GTSP instance
    :param tour: the tour
    :param cluster_tour: the cluster tour
    :return: the refined tour and the refined cluster tour
    """
    modified = True
    while modified:
        modified = False
        best_tour = tour
        best_cluster_tour = cluster_tour
        best_d = gtsp.cost_time_ratio(tour)[0]
        # find for every poi
        for c_alpha_ind in range(len(cluster_tour) - 2):
            for c_gamma_ind in range(c_alpha_ind + 2, len(cluster_tour)):
                
                c_beta_ind = (c_alpha_ind + 1) % len(cluster_tour)
                c_delta_ind = (c_gamma_ind + 1) % len(cluster_tour)
                prec_alpha = tour[(c_alpha_ind - 1) % len(tour)]
                next_beta = tour[(c_beta_ind + 1) % len(tour)]
                prec_gamma = tour[(c_gamma_ind - 1) % len(tour)]
                next_delta = tour[(c_delta_ind + 1) % len(tour)]
                u,v,w,z = None, None, None, None
                min_d_uw = np.inf
                min_d_vz = np.inf
                for a in gtsp.clusters[cluster_tour[c_alpha_ind]]:
                    for b in gtsp.clusters[cluster_tour[c_gamma_ind]]:
                        d_ia = np.linalg.norm(gtsp.points[prec_alpha] - gtsp.points[a])
                        d_ab = np.linalg.norm(gtsp.points[a] - gtsp.points[b])
                        d_bh = np.linalg.norm(gtsp.points[b] - gtsp.points[prec_gamma])
                        if d_ia + d_ab + d_bh < min_d_uw:
                            min_d_uw = d_ia + d_ab + d_bh
                            u = a
                            w = b
                for a in gtsp.clusters[cluster_tour[c_beta_ind]]:
                    for b in gtsp.clusters[cluster_tour[c_delta_ind]]:
                        d_ja = np.linalg.norm(gtsp.points[next_beta] - gtsp.points[a])
                        d_ab = np.linalg.norm(gtsp.points[a] - gtsp.points[b])
                        d_bk = np.linalg.norm(gtsp.points[b] - gtsp.points[next_delta])
                        if d_ja + d_ab + d_bk < min_d_vz:
                            min_d_vz = d_ja + d_ab + d_bk
                            v = a
                            z = b
                # Compute the new length of the tour
                new_d = 0
                new_tour = tour.copy()
                new_tour[c_alpha_ind] = u
                new_tour[c_beta_ind] = w
                new_tour[c_gamma_ind] = v
                new_tour[c_delta_ind] = z
                new_cluster_tour = cluster_tour.copy()
                new_cluster_tour[c_beta_ind] = cluster_tour[c_gamma_ind]
                new_cluster_tour[c_gamma_ind] = cluster_tour[c_beta_ind]
                new_d = gtsp.cost_time_ratio(new_tour)[0]
                if new_d < best_d:
                    best_tour = new_tour
                    best_cluster_tour = new_cluster_tour
                    best_d = new_d
                    modified = True
        if modified:
            tour = best_tour
            cluster_tour = best_cluster_tour
    return tour, cluster_tour

def RP1_procedure_paral(gtsp: GTSP, tour: list[int], cluster_tour: list[int]) -> tuple[list[int], list[int]]:
    """
    Apply the RP1 procedure to refine a tour (2-opt generalization)
    :param gtsp: the GTSP instance
    :param tour: the tour
    :param cluster_tour: the cluster tour
    :return: the refined tour and the refined cluster tour
    """
    modified = True
    cur_sol = DGTSPPoint(gtsp, tour, cluster_tour)
    while modified:
        modified = False
        neighbors = cur_sol.neighbors_2opt()
        for neighbor in neighbors:
            if neighbor.value[0] < cur_sol.value[0]:
                cur_sol = neighbor
                modified = True
    return cur_sol.tour, cur_sol.cluster_tour
if __name__ == "__main__":
    name = "pr107"
    time_start = time()
    inst = GTSP.from_file(f"TSP/Instances_TSP/{name}.tsp")
    sol_f = farthest_insertion(inst)
    tour_f, cluster_tour_f = sol_f.tour, sol_f.cluster_tour
    sol_n = nearest_insertion(inst)
    tour_n, cluster_tour_n = sol_n.tour, sol_n.cluster_tour
    cost_f = inst.cost_time_ratio(tour_f)
    cost_n = inst.cost_time_ratio(tour_n)
    if cost_f[0] < cost_n[0]:
        print("Farthest insertion is better")
        inst.plot_tour(tour_f, f"tmp/tour_heur_pre_{name}.png")
        print(cost_f)
    else:
        print("Nearest insertion is better")
        inst.plot_tour(tour_n, f"tmp/tour_heur_pre_{name}.png")
        print(cost_n)
    print(f"Time: {time()-time_start}")
    tour_frp1, cluster_tour_frp1 = RP1_procedure_paral(inst, tour_f, cluster_tour_f)
    tour_nrp1, cluster_tour_nrp1 = RP1_procedure_paral(inst, tour_n, cluster_tour_n)
    cost_frp1 = inst.cost_time_ratio(tour_frp1)
    cost_nrp1 = inst.cost_time_ratio(tour_nrp1)
    if cost_frp1[0] < cost_nrp1[0]:
        print("Farthest insertion is better")
        inst.plot_tour(tour_frp1, f"tmp/tour_heur_{name}.png")
        print(cost_frp1)
    else:
        print("Nearest insertion is better")
        inst.plot_tour(tour_nrp1, f"tmp/tour_heur_{name}.png")
        print(cost_nrp1)
    print(f"Time: {time()-time_start}")
