import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

curr_dir = os.path.dirname(__file__)
pol_dir = os.path.join(curr_dir, "polblogs")

class PolBlog:

    def __init__(self):
        print("Reading in PolBlog data...")
        filepath = os.path.join(pol_dir, "polblogs.gml")
        self.graph = nx.read_gml(filepath)
        self.graph = self.graph.to_undirected()
        self.filter_out_low_degree()
        self.pos = self.fruchterman_reingold()
        print("Done")


    def filter_out_low_degree(self, min_degree=2):
        low_degree_nodes = [node for node, degree in dict(self.graph.degree).items() if degree < min_degree]
        self.graph.remove_nodes_from(low_degree_nodes)


    def political_partition(self):
        politics_arr = []
        for node_label, attributes in self.graph.nodes.items():
            politics_arr.append(attributes["value"])

        return politics_arr

    
    def fruchterman_reingold(self):
        return nx.spring_layout(self.graph)


    def k_means_centres(self, k=2):
        print("Performing k-means...")
        # nodes_by_degree_desc = sorted(self.graph.degree, key = lambda x: x[1], reverse=True)
        node_pos_arr = list(self.pos.values())
        kmeans = KMeans(n_clusters=k).fit(node_pos_arr) # can use flag random_state=0 to intialise seed
        return kmeans.cluster_centers_



    def draw(self):
        print("Drawing network...")
        politics_arr = self.political_partition()
        colors = ["blue" if value == 0 else "red" for value in politics_arr]
        cluster_centres = self.k_means_centres()

        nx.draw_networkx(self.graph, pos=self.pos, with_labels=False, node_size=5, width=0.1, node_color=colors)

        plt.title("Political blogs (2004 US Election)")

        # labels
        plt.scatter([], [], c="blue", label="Democrat")
        plt.scatter([], [], c="red", label="Republican")

        for centre in cluster_centres:
            plt.scatter(centre[0], centre[1], c="yellow", s=100, marker="x", zorder=100)

        plt.legend()
        plt.show()