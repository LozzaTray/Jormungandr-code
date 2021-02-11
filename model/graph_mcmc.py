import networkx as nx
from graph_tool import Graph as GT_Graph
from graph_tool.all import graph_draw
from graph_tool.inference import minimize_blockmodel_dl
import matplotlib.pyplot as plt
## version focal seems to be winner


class Graph_MCMC:

    def __init__(self, edges):
        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        self.N = len(vertex_set)
        self.M = len(edges)

        self.G = GT_Graph(directed=False)
        self.G.add_edge_list(edges)
        
        # initialise empty state
        self.state = None

        print("Initialised graph with N={} nodes and M={} edges".format(self.N, self.M))

    
    def partition(self, B_min=None, B_max=None, degree_corrected=True):
        """Performs MCMC algorithm to minimise description length (DL)"""
        print("Performing inference...")
        self.state = minimize_blockmodel_dl(self.G, B_min=B_min, B_max=B_max, deg_corr=degree_corrected, verbose=True)
        print("Done")
        return self.state.entropy()


    def draw(self, output=None):
        output = self.gen_output_path(output)
        if self.state is not None:
            print("Drawing state partition")
            self.state.draw(output=output)
        else:
            print("No state partition detected >> draw default graph")
            graph_draw(self.G, vertex_text=self.G.vertex_index, output=output)

    
    def plot_matrix(self):
        if self.state is not None:
            print("Drawing block adjacency matrix")
            block_adjacency_matrix = self.state.get_matrix()
            plt.matshow(block_adjacency_matrix.todense())
            plt.title("Block Adjacency Matrix")
            plt.colorbar()
            plt.show()
        else:
            print("No state partition detected >> cannot draw matrix")


    def gen_output_path(self, filename):
        ## valid extensions: .pdf, .png, .svg
        if filename is not None:
            return "output/" + filename

