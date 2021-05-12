import numpy as np
# version focal seems to be winner
from graph_tool import Graph as GT_Graph
# X-server must be running else import will timeout
from graph_tool.draw import graph_draw
from graph_tool.inference import minimize_blockmodel_dl, mcmc_equilibrate, PartitionModeState
from graph_tool.collection import data, ns
import matplotlib.pyplot as plt
from inference.softmax import SoftmaxNeuralNet, from_values_to_one_hot
from data.utils import get_misc_path
from tqdm import tqdm
import os

curr_dir = os.path.dirname(__file__)
output_dir = os.path.join(curr_dir, "..", "output")


class Graph_MCMC:


    def __init__(self):
        """Initialises empty graph"""
        self.G = GT_Graph(directed=False)
        
        # initialise empty state
        self.state = None
        self.vertex_block_counts = None
        self.B_max = None
        self.relabelled_vertices = None

        # treat prior on b as uniform
        self.entropy_args = {"partition_dl": False}
        self.mcmc_args = {"entropy_args": self.entropy_args}



    def read_from_edges(self, edges):
        """Initialises graph based on edges"""
        origin_vertices = set([edge[0] for edge in edges])
        destin_vertices = set([edge[1] for edge in edges])

        vertex_set = origin_vertices.union(destin_vertices)
        
        # constants
        N = len(vertex_set)
        M = len(edges)

        self.relabelled_vertices = self.G.add_edge_list(edges, hashed=True)

        print("Initialised graph with N={} nodes and M={} edges".format(N, M))

    
    def read_from_file(self, filename):
        filename = get_misc_path(filename)
        self.G.load(filename)

    
    def read_from_gt(self, dataset_name):
        self.G = data[dataset_name]
        self.G.set_directed(False)
        print("Vertex props: " + str(self.G.vertex_properties.keys()))

    
    def read_from_ns(self, dataset_name):
        self.G = ns[dataset_name]
        self.G.set_directed(False)
        print("Vertex props: " + str(list(self.G.vertex_properties.keys())))


    def filter_out_low_degree(self, min_degree):
        """Removes all vertices with degree strictly less than min_degree"""
        vertices = self.G.get_vertices()
        degree_arr = self.G.get_total_degrees(vertices)
        
        remove_arr = []
        for idx, degree in enumerate(degree_arr):
            if degree < min_degree:
                remove_arr.append(idx)
        
        self.G.remove_vertex(remove_arr)

    
    def filter_edges(self, property_name, value_to_keep):
        prop_map = self.G.edge_properties[property_name]
        filter_arr = [value == value_to_keep for value in prop_map.get_array()]
        filter_map = self.G.new_edge_property("bool", filter_arr)
        self.G.set_edge_filter(filter_map)


    def add_ego_node(self):
        vertices = self.G.get_vertices()
        v = self.G.add_vertex()
        edges = []
        for dest in vertices:
            edges.append((v, dest))

        self.G.add_edge_list(edges)


    def get_vertex_list(self):
        """Returns external view of vertices"""
        if self.relabelled_vertices is not None:
            return self.relabelled_vertices.get_array()
        else:
            return self.G.get_vertices()


    def add_property(self, name, value_type, value_sequence):
        vertex_prop = self.G.new_vertex_property(value_type, value_sequence)
        self.G.vertex_properties[name] = vertex_prop # add to graph


    def remove_property(self, name):
        if name in self.G.vertex_properties:
            del self.G.vertex_properties[name]
            return True
        return False


    def convert_props_to_flags(self):
        properties = self.G.vertex_properties
        vertices = self.G.get_vertices()
        property_names = list(properties.keys())

        for name in property_names:
            value_map = properties[name]
            value_type = value_map.value_type()
            if value_type == "string":
                values = [value_map[vertex] for vertex in vertices]
                distinct_features = set(values)

                distinct_features.discard("") # remove empty string if it exists

                for feature in sorted(distinct_features):
                    bool_arr = []
                    for value in values:
                        if value == feature:
                            bool_arr.append(True)
                        else:
                            bool_arr.append(False)
                        
                        self.add_property(feature, "bool", bool_arr)
            
                self.remove_property(name)

            elif value_type.startswith("int"):
                self.convert_to_flags(name)
            
            elif value_type == "bool":
                pass

            elif value_type == "float": 
                pass
            else: # vector
                self.remove_property(name)


    def convert_to_flags(self, prop_name, new_prop_name=""):
        vertices = self.G.get_vertices()

        if new_prop_name == "":
            new_prop_name = prop_name + "-"

        if prop_name in self.G.vertex_properties:
            value_map = self.G.vertex_properties[prop_name]

            values = [value_map[vertex] for vertex in vertices]
            distinct_values = set(values)

            for chosen_value in sorted(distinct_values):
                bool_arr = []
                for value in values:
                    if value == chosen_value:
                        bool_arr.append(True)
                    else:
                        bool_arr.append(False)
                    
                    self.add_property(new_prop_name + str(chosen_value), "bool", bool_arr)
        
            self.remove_property(prop_name)


    def partition(self, B_min=None, B_max=None, degree_corrected=True):
        """
        Performs MCMC algorithm to minimise description length (DL)
        returns partition array
        """
        print("Performing inference...")
        self.state = minimize_blockmodel_dl(self.G, B_min=B_min, B_max=B_max, deg_corr=degree_corrected, mcmc_args=self.mcmc_args, verbose=True)
        print("Done")
        return self.state.get_blocks()

    
    def mcmc(self, num_iter, verbose=False):
        """
        Performs mcmc sampling of posterior on blocks
        returns: B_max - the highest number of blocks across samples
        """
        bs = [] # collect some partitions

        def collect_partitions(s):
            bs.append(s.b.a.copy())

        for i in tqdm(range(0, num_iter)):
                dS, nattempts, nmoves = self.state.mcmc_sweep(niter=1, d=0.00, entropy_args=self.entropy_args)
                collect_partitions(self.state)
                if verbose and i % 10 == 0:
                    print("i: {}, dS: {}, nattempts: {}, nmoves: {}".format(i, dS, nattempts, nmoves))

        # mcmc_equilibrate(self.state, force_niter=num_iter, callback=collect_partitions, mcmc_args={"d": 0.00}, verbose=verbose)
        # BUG: parameter not passed through correctly

        # Disambiguate partitions and obtain marginals
        pmode = PartitionModeState(bs, converge=True, relabel=True)
        pv = pmode.get_marginal(self.G)

        # Now the node marginals are stored in property map pv. We can
        # visualize them as pie charts on the nodes:
        self.vertex_block_counts = pv
        self.B_max = pmode.get_B()
        return self.B_max


    def generate_posterior(self):
        """
        return Y: (N x B) matrix of posterior probabilities
        Y[n, b] = Prob vertex n belongs to block b
        """
        vertices = self.G.get_vertices()
        N = len(vertices)
        B = self.B_max

        posterior_probs = np.zeros((N, B))

        for idx, vertex_id in enumerate(vertices):
            counts = self.vertex_block_counts[vertex_id]
            total = np.sum(counts)
            probs = counts / total
            b = len(probs)

            posterior_probs[idx, 0:b] = probs[:]

        # block_counts = np.sum(posterior_probs, axis=0)
        # indices = np.argsort(block_counts)[::-1]

        # sorted_posterior_probs = np.zeros((N, B))
        # sorted_posterior_probs[:, :] = posterior_probs[:, indices]

        # return sorted_posterior_probs
        return posterior_probs

    
    def generate_feature_matrix(self):
        """
        return X: (N x D) matrix of node features
        X[n, d] = feature d of vertex n
        """
        properties = self.G.vertex_properties
        D = len(properties)

        vertices = self.G.get_vertices()
        
        N = len(vertices)
        X = np.empty((N, D))

        for prop_index, value_map in enumerate(properties.values()):
            for vertex_index, vertex_id in enumerate(vertices):
                X[vertex_index, prop_index] = value_map[vertex_id]
        
        return X

    
    def get_feature_names(self):
        """returns array of feature names"""
        properties = self.G.vertex_properties
        return [key.replace("\x00", "-") for key in properties.keys()]

    
    def sample_classifier_marginals(self, num_iter, step_scaling=1, sigma=1, verbose=False):
        if self.vertex_block_counts is None:
            print("Cannot sample without marginals")
        else:
            X = self.generate_feature_matrix()
            Y = self.generate_posterior()

            D = X.shape[1]
            B = Y.shape[1]

            classifier = SoftmaxNeuralNet(layers_size=[D, B], sigma=sigma)
            classifier.sgld_initialise()

            for i in tqdm(range(0, num_iter)):
                cost = classifier.sgld_iterate(X=X, Y=Y, step_scaling=step_scaling)
                if verbose and i % 10 == 0:
                    print("i: {}, cost: {}".format(i, cost))

            return classifier


    
    def sample_classifier_mcmc(self, num_iter, verbose=False):
        if self.state is None:
            print("No state partition detected >> ABORT")
        else:
            properties = self.G.vertex_properties
            D = len(properties)

            B = self.state.get_nonempty_B()
            vertices = self.G.get_vertices()
            N = len(vertices)

            X = self.generate_feature_matrix()

            classifier = SoftmaxNeuralNet(layers_size=[D, B])

            def sgld_iterate(state):
                blocks = state.get_blocks()
                Y = np.empty(N)

                for vertex_index, vertex_id in enumerate(vertices):
                    Y[vertex_index] = blocks[vertex_id]

                Y = from_values_to_one_hot(Y)
                cost = classifier.sgld_iterate(X=X, Y=Y)
                return cost

            classifier.sgld_initialise()

            # mcmc_equilibrate(self.state, force_niter=num_iter, callback=sgld_iterate, verbose=verbose)
            for i in tqdm(range(0, num_iter)):
                dS, nattempts, nmoves = self.state.multiflip_mcmc_sweep(niter=1, psplit=0)
                cost = sgld_iterate(self.state)
                if verbose and i % 10 == 0:
                    print("i: {}, dS: {}, nattempts: {}, nmoves: {}, cost: {}".format(i, dS, nattempts, nmoves, cost))

            return classifier

        

    def draw(self, output=None):
        output = self.gen_output_path(output)            
        if self.state is not None:
            if self.vertex_block_counts is not None:
                print("Drawing soft partition")
                self.state.draw(vertex_shape="pie", vertex_pie_fractions=self.vertex_block_counts, output=output)
            else:
                print("Drawing hard state partition")
                self.state.draw(output=output)
        else:
            print("No state partition detected >> draw default graph")
            graph_draw(self.G, output=output)

    
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

    
    def train_map_classifier(self):
        if self.state is None:
            print("No state partition detected >> ABORT")
        else:
            B = self.state.get_nonempty_B()
            vertices = self.G.get_vertices()
            N = len(vertices)

            X = self.generate_feature_matrix()
            D = X.shape[1]

            blocks = self.state.get_blocks() # dictionary: vertex -> block_index
            Y = np.empty(N)

            for vertex_index, vertex_id in enumerate(vertices):
                Y[vertex_index] = blocks[vertex_id]

            classifier = SoftmaxNeuralNet(layers_size=[D, B])
            classifier.fit(X, Y)

            return classifier

    
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


    def plot_posterior_props(self):
        if self.vertex_block_counts is not None:
            posterior = self.generate_posterior()
            block_counts = np.sum(posterior, axis=0)
            x = np.arange(0, len(block_counts), 1)
            plt.bar(x, block_counts)
            plt.title("Block Membership Counts")
            plt.ylabel("Cumulative probability sum")
            plt.xlabel("Block index")
            plt.show()
        else:
            raise Exception("Vertex block counts not initialised")


    def gen_output_path(self, filename):
        ## valid extensions: .pdf, .png, .svg
        if filename is not None:
            return os.path.join(output_dir, filename)
        return None

