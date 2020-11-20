import numpy as np
from tqdm import tqdm
import math

import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, edges):

        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        self.N = len(vertex_set)
        self.M = len(edges)

        # edges
        self.edges_raw = edges

        # relabel vertices with an index
        self.index_to_vertex = {}
        self.vertex_to_index = {}

        for index, vertex in enumerate(vertex_set):
            self.index_to_vertex[index] = vertex
            self.vertex_to_index[vertex] = index

        self.adjacency_list = [set() for v in range(0, self.N)]
        self.adjacency_matrix = np.zeros((self.N, self.N))

        for edge in edges:
            a = self.vertex_to_index.get(edge[0])
            b = self.vertex_to_index.get(edge[1])

            self.adjacency_list[a].add(b)
            self.adjacency_list[b].add(a)

            self.adjacency_matrix[a, b] += 1
            self.adjacency_matrix[b, a] += 1

        self.assignments = None
        print("Initialised graph with N={} nodes and M={} edges".format(self.N, self.M))


    def abp(self):
        # implement for r = 3
        r = 3
        T = math.floor(math.log2(self.N)) # num_iters - r

        y = np.zeros((T, self.N, self.N)) # y[t, i, j] is message at time t from i -> j
        z = np.zeros((T, self.N, self.N)) # z[t, i, j] is normalised message at t from i -> j
        
        # random initial setting
        y[r-1, :, :] = np.random.normal(size=(self.N, self.N))

        for t in tqdm(range(r, T)):
            s = np.sum(y[t-1, :, :]) / (2 * self.M)
            z[t-1, :, :] = y[t-1, :, :] - s

            for v in range(0, self.N):
                vd_set = self.adjacency_list[v]
                
                for vd in vd_set:
                    y_vd_to_v = 0
                    vdd_set = self.adjacency_list[vd].difference({v})

                    for vdd in vdd_set:
                        y_vd_to_v += z[t-1, vdd, vd]

                        if vdd in vd_set:
                            # part of 3-cycle
                            y_vd_to_v -= z[t-3, vdd, v]
                    
                    y[t, vd, v] = y_vd_to_v

        output = np.zeros(self.N)
        for v in range(0, self.N):
            vd_set = self.adjacency_list[v]

            for vd in vd_set:
                output[v] += y[T-1, vd, v]

        self.output = output
        self.assignments = (output > 0).astype(int)

        community_one_ratio = np.sum(self.assignments) / self.N
        print("Partitioned such that p={} in community 1".format(community_one_ratio))


    def draw(self):
        node_color = self.build_color_arr()
        node_pos = self.gen_layout()
        G = nx.Graph()
        G.add_nodes_from(self.vertex_to_index.keys())
        G.add_edges_from(self.edges_raw)
        nx.draw_networkx(G, pos=node_pos, with_labels=False, node_size=2, node_color=node_color, width=0.1) 
        plt.show()


    def build_color_arr(self):
        if self.assignments is not None:
            colors = []
            for vertex, index in self.vertex_to_index.items():
                if self.assignments[index] == 1:
                    colors.append("blue")
                else:
                    colors.append("red")
            return colors
        else:
            return "orange"


    def gen_layout(self):
        pos = self.vertex_to_index.copy()
        c = 2
        s = 1
        for vertex, index in self.vertex_to_index.items():
            random = s * np.random.normal(size=(2))
            if self.assignments[index] == 1:
                pos[vertex] = (c + random[0], random[1])
            else:
                pos[vertex] = (-c + random[0], random[1])

        return pos









