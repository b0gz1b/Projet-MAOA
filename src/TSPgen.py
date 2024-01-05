import numpy as np
import networkx as nx
from TSP import TSP
from preprocessing import affinity_propagation, k_means, plot_km
import matplotlib.pyplot as plt
# try:
#     from concorde.tsp import TSPSolver
# except ImportError:
#     pass

def GTSP_to_ATSP(points, labels):
    """
    Transforms a GTSP instance into a TSP instance.
    :param gtsp_instance: the GTSP instance
    :return: the TSP instance
    """
    # Compute M, the sum of all the weights of the edges between the points
    M = 100
    labels = np.array(labels)
    for i in range(len(points)):
        for j in range(len(points)):
            if labels[i] != labels[j]:
                M += np.linalg.norm(points[i] - points[j])
    G = nx.DiGraph()
    G.add_nodes_from(range(len(points)))
    # Add edges to form a cycle in each cluster
    for cluster in range(len(set(labels))):
        cluster_points_indices = np.where(labels == cluster)[0]
        for i in range(len(cluster_points_indices)):
            G.add_edge(cluster_points_indices[i], cluster_points_indices[(i + 1) % len(cluster_points_indices)], weight=0)
    # Add edges between every points in different clusters
    for cluster1 in range(len(labels)):
        cluster1_points_indices = np.where(labels == cluster1)[0]
        for cluster2 in range(len(labels)):
            if cluster1 != cluster2:
                cluster2_points_indices = np.where(labels == cluster2)[0]
                for i,cluster1_point_indice in enumerate(cluster1_points_indices):
                    for cluster2_point_indice in cluster2_points_indices:
                        # The weight of the edge between point1 and point2 the cost of the edge formed by the next node in the cycle of cluster1 after point1 and point2 plus M
                        G.add_edge(cluster1_point_indice, cluster2_point_indice, weight = M + np.linalg.norm(
                            points[cluster1_points_indices[(i + 1) % len(cluster1_points_indices)]] - points[cluster2_point_indice]))
    # Complete the complete graph
    return G

def ATSP_to_TSP(G):
    """
    Transforms an ATSP instance into a TSP instance.
    :param atsp_instance: the ATSP instance
    :return: the TSP instance
    """
    M = 100
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    # To double the size, each of the nodes in the graph is duplicated, 
    # creating a second ghost node, linked to the original node with a "ghost" 
    # edge of very low (possibly negative) weight, here denoted −w. 
    # (Alternatively, the ghost edges have weight 0, and weight w is 
    # added to all other edges.) The original 3×3 matrix shown above 
    # is visible in the bottom left and the transpose of the original 
    # in the top-right. Both copies of the matrix have had their diagonals 
    # replaced by the low-cost hop paths, represented by −w. 
    # In the new graph, no edge directly links original nodes and 
    # no edge directly links ghost nodes.
    # Add the ghost nodes
    for node in G.nodes():
        H.add_node(node + len(G.nodes()))
    
    # Add the edges between the original nodes and the ghost nodes
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            H.add_edge(node, neighbor + len(G.nodes()), weight=G[node][neighbor]['weight'] + 100)
            M += G[node][neighbor]['weight']
    # Add the ghost edges
    for node in G.nodes():
        H.add_edge(node, node + len(G.nodes()), weight=0)
    # Complete the complete graph
    for i in range(len(H.nodes())):
        for j in range(len(H.nodes())):
            if not H.has_edge(i, j):
                H.add_edge(i, j, weight=M)
    return H

def TSP_to_file(G, file_path):
    """
    Writes a TSP instance to a file.
    :param tsp_instance: the TSP instance
    :param file_path: the file path
    """
    with open(file_path, 'w') as file:
        file.write("NAME: " + file_path.split("/")[-1] + "\n")
        file.write("TYPE: TSP\n")
        file.write("COMMENT: " + file_path.split("/")[-1] + "\n")
        file.write("DIMENSION: " + str(len(G.nodes())) + "\n")
        file.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        file.write("DISPLAY_DATA_TYPE: NO_DISPLAY\n")
        file.write("EDGE_WEIGHT_SECTION\n")
        for i in range(len(G.nodes())):
            for j in range(len(G.nodes())):
                file.write(str(int(G[i][j]['weight'])) + " ")
            file.write("\n")
        file.write("EOF\n")

def TSPtour_to_ATSPtour(tour):
    """
    Transforms a TSP tour into an ATSP tour.
    :param tour: the TSP tour
    :param atsp_instance: the ATSP instance
    :return: the ATSP tour
    """
    tour_even = tour[::2]
    tour_odd = tour[1::2]
    return tour_even if tour_even[0] < tour_odd[0] else tour_odd

def ATSPtour_to_GTSPtour(tour, labels):
    """
    Transforms an ATSP tour into a GTSP tour.
    :param tour: the ATSP tour
    :param gtsp_instance: the GTSP instance
    :return: the GTSP tour
    """
    gtsp_tour = []
    # Find the edges between the clusters
    for i in range(len(tour)):
        if labels[tour[i]] != labels[tour[(i + 1) % len(tour)]]:
            gtsp_tour.append(tour[(i + 1) % len(tour)])
    return gtsp_tour



def plot_tour(points, labels, tour, file_path=None):
    """
    Plots a tour.
    :param points: the points
    :param labels: the labels of the points
    :param tour: the tour
    """
    fig, ax = plt.subplots()
    # K-means plot
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    # add labels to the points
    for i in range(len(points)):
        ax.annotate(i, (points[i, 0], points[i, 1]))
    # plot the tour edges
    for i in range(len(tour)):
        ax.plot([points[tour[i], 0], points[tour[(i + 1) % len(tour)], 0]], [points[tour[i], 1], points[tour[(i + 1) % len(tour)], 1]], color='black', alpha=0.5, linewidth=0.5, linestyle='--')
    ax.set_title('Tour')
    if file_path is None:
        plt.savefig("tmp/tour.png")
    else:    
        plt.savefig(file_path)
if __name__ == '__main__':
    inst = TSP.from_file("TSP/Instances_TSP/PLS_21.tsp")
    cluster_centers_km, labels_km = k_means(inst.points, n_clusters=4, n_init=100)
    # labels_km = [0,3,0,3,1,1,2,1,0,2,2,0,3,1,3,1,1,1,2,0,0]
    # plot_km(points, labels_km, cluster_centers_km)
    G = GTSP_to_ATSP(inst.points, labels_km)
    plt.figure()
    nx.draw(G, pos=inst.points, with_labels=True, font_weight='bold')
    plt.savefig("tmp/G.png")
    H = ATSP_to_TSP(G)
    # Draw the new nodes added to H compared to G
    plt.figure()
    nx.draw(H, pos=nx.spring_layout(H), with_labels=True, nodelist=range(len(G.nodes()), len(H.nodes())), node_color='r', font_weight='bold')
    plt.savefig("tmp/H.png")
    TSP_to_file(H, "TSP/Instances_TSP/PLS_21_transf.tsp")
    # solver = TSPSolver.from_tspfile("TSP/Instances_TSP/PLS_21_transf.tsp")
    # solution = solver.solve()
    # print(solution.tour)
    # tour_atsp = TSPtour_to_ATSPtour(solution.tour)
    # print(tour_atsp)
    # # print the clusters of the tour
    # for i in range(len(tour_atsp)):
    #     print(labels_km[tour_atsp[i]], end=" ")
    # print()
    # tour_gtspt = ATSPtour_to_GTSPtour(tour_atsp, labels_km)
    # print(tour_gtspt)
    # plot_tour(inst.points, labels_km, tour_atsp, "tmp/tour_atsp.png")
    # plot_tour(inst.points, labels_km, tour_gtspt, "tmp/tour_gtspt.png")


    
