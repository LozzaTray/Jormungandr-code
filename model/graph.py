import numpy as np
from tqdm import tqdm
import math


class Graph:

    def __init__(self, edges):

        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        self.N = len(vertex_set)
        self.M = len(edges)

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

        print(np.mean(np.abs(output)))
        self.assignments = (output > 0).astype(int)
        print(self.assignments)






