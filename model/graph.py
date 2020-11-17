import numpy as np


class Graph:

    def __init__(self, edges):

        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        self.N = len(self.vertex_set)
        self.M = len(edges)

        # relabel vertices with an index
        self.index_to_vertex = {}
        self.vertex_to_index = {}

        for index, vertex in enumerate(vertex_set):
            self.index_to_vertex[index] = vertex
            self.vertex_to_index[vertex] = index

        self.adjacency_list = [[] for v in range(0, self.N)]
        self.adjacency_matrix = np.zeros((self.N, self.N))

        for edge in edges:
            a = self.vertex_to_index.get(edge[0])
            b = self.vertex_to_index.get(edge[1])

            self.adjacency_list[a].append(b)
            self.adjacency_list[b].append(a)

            self.adjacency_matrix[a, b] += 1
            self.adjacency_matrix[b, a] += 1


    def abp(self):
        r = 3 # max length of cycle to correct for
        T = 1000 # num_iters

        y = np.zeros((T, self.N, self.N)) # y[t, i, j] is message at time t from i -> j
        z = np.zeros((T, self.N, self.N)) # z[t, i, j] is normalised message at t from i -> j
        
        # random initial setting
        y[r-1, :, :] = np.random.normal(size=(self.N, self.N))

        for t in range(r, T):
            for v in range(0, self.N):
                adjacent_vertices = self.adjacency_list[v]

                for vd in adjacent_vertices:
                    y_vd_to_v = 




