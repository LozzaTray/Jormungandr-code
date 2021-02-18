import numpy as np
from graph_tool import Graph as GT_Graph
from graph_tool.all import graph_draw
from graph_tool.inference import minimize_blockmodel_dl
import matplotlib.pyplot as plt
from graph_tool.all import BlockState
from inference.softmax import SoftmaxNeuralNet
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


    def get_vertex_list(self):
        return self.G.get_vertices()


    def add_property(self, name, value_type, value_sequence):
        vertex_prop = self.G.new_vertex_property(value_type, value_sequence)
        self.G.vertex_properties[name] = vertex_prop # add to graph


    def partition(self, B_min=None, B_max=None, degree_corrected=True):
        """
        Performs MCMC algorithm to minimise description length (DL)
        returns partition array
        """
        print("Performing inference...")
        self.state = minimize_blockmodel_dl(self.G, B_min=B_min, B_max=B_max, deg_corr=degree_corrected, verbose=True)
        print("Done")
        return self.state.get_blocks()


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

    
    def train_feature_classifier(self):
        if self.state is None:
            print("No state partition detected >> ABORT")
        else:
            properties = self.G.vertex_properties
            D = len(properties)

            B = self.state.get_B()
            vertices = self.G.get_vertices()
            N = len(vertices)

            X = np.empty((N, D))

            for prop_index, value_map in enumerate(properties.values()):
                for vertex_index, vertex_id in enumerate(vertices):
                    X[vertex_index, prop_index] = value_map[vertex_id]

            blocks = self.state.get_blocks() # dictionary: vertex -> block_index
            Y = np.empty(N)

            for vertex_index, vertex_id in enumerate(vertices):
                Y[vertex_index] = blocks[vertex_id]

            classifier = SoftmaxNeuralNet(layers_size=[B])
            classifier.fit(X, Y)

            classifier.plot_final_weights(list(properties.keys()))
            classifier.plot_cost()

    
    def plot_community_property_fractions(self):
        if self.state is not None:
            properties = self.G.vertex_properties
            num_properties = len(properties)

            blocks = self.state.get_blocks()
            B = self.state.get_B()
            vertices = self.G.get_vertices()
            block_counts = np.zeros(B)

            for v in vertices:
                block_index = blocks[v]
                block_counts[block_index] += 1

            width = 0.8 / num_properties
            idx = 0

            for prop_name, value_map in properties.items():
                prop_counts = np.zeros(B)

                for v in vertices:
                    block_index = blocks[v]

                    if value_map[v]:
                        prop_counts[block_index] += 1

                prop_fractions = np.divide(prop_counts, block_counts)
                x = np.array(range(0, B)) + (width * idx)
                plt.bar(x, prop_fractions, width=width, label=prop_name)
                idx += 1
            
            plt.title("Community fractions in detected blocks")
            plt.xlabel("Inferred block")
            plt.ylabel("Fraction of vertices with given feature")
            plt.legend()
            plt.show()
        else:
            print("No state partition detected >> cannot draw prop fractions")


    def gen_output_path(self, filename):
        ## valid extensions: .pdf, .png, .svg
        if filename is not None:
            return "output/" + filename

