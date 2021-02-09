import networkx as nx
from graph_tool import Graph as GT_Graph
from graph_tool.all import graph_draw
## version focal seems to be winner


class Graph_MCMC:

    def __init__(self, edges, num_blocks, degree_corrected=False):
        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        self.N = len(vertex_set)
        self.M = len(edges)
        self.K = num_blocks

        self.G = GT_Graph(directed=False)
        self.G.add_edge_list(edges)

        print("Initialised graph with N={} nodes and M={} edges".format(self.N, self.M))
        if degree_corrected:
            print("Degree corrected model")

    
    def draw(self):
        graph_draw(self.G, vertex_text=self.G.vertex_index, output="test.pdf")
