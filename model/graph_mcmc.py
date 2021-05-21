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


def gen_output_path(filename):
    ## valid extensions: .pdf, .png, .svg
    if filename is not None:
        return os.path.join(output_dir, filename)
    return None


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


    # print helpers
    def print_info(self):
        N = self.G.num_vertices()
        E = self.G.num_edges()
        D = len(self.get_feature_names())
        print("Graph with N={} nodes, E={} edges and D={} vertex features for training".format(N, E, D))

    
    def print_props(self):
        print("All vertex props: " + str(set(self.G.vertex_properties.keys())))
        print("Training vertex props: " + str(self.get_feature_names()))

    
    def list_props(self):
        self.G.list_properties()


    # read helpers
    def read_from_edges(self, edges):
        """Initialises graph based on edges"""
        self.relabelled_vertices = self.G.add_edge_list(edges, hashed=True)

    def read_from_file(self, filename):
        filename = get_misc_path(filename)
        self.G.load(filename)

    
    def read_from_gt(self, dataset_name):
        self.G = data[dataset_name]
        self.G.set_directed(False)

    
    def read_from_ns(self, dataset_name):
        self.G = ns[dataset_name]
        self.G.set_directed(False)

    
    # save helpers
    def save_to_file(self, name):
        filename = gen_output_path(name)
        self.G.save(filename, fmt="gt")


    # filter
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
        internal_vertices = self.G.get_vertices()
        if self.relabelled_vertices is not None:
            return [self.relabelled_vertices[vertex] for vertex in internal_vertices]
        else:
            return internal_vertices


    # property methods
    def add_property(self, name, value_type, value_sequence):
        vertex_prop = self.G.new_vertex_property(value_type, value_sequence)
        self.G.vertex_properties[name] = vertex_prop # add to graph


    def remove_property(self, name):
        if name in self.G.vertex_properties:
            del self.G.vertex_properties[name]
            return True
        return False

    
    def rename_property(self, old_name, new_name):
        if old_name in self.G.vertex_properties:
            self.G.vertex_properties[new_name] = self.G.vertex_properties[old_name]
            del self.G.vertex_properties[old_name]
            return True
        return False


    def convert_props_to_flags(self):
        vertices = self.G.get_vertices()
        property_names = self.get_feature_names()

        for name in property_names:
            value_map = self.get_property_map(name)
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

        if prop_name in self.get_feature_names():
            value_map = self.get_property_map(prop_name)

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

    
    def get_property_map(self, prop_name):
        property_map = self.G.vertex_properties[prop_name]
        return property_map

    
    def get_feature_names(self):
        """returns array of feature names"""
        properties = self.G.vertex_properties
        # return [key.replace("\x00", "-") for key in properties.keys()]
        names = list(properties.keys())
        feature_names = [name for name in names if name.startswith("_") == False]
        return feature_names


    # sampling methods
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
        returns: av_entropy_per_node - average netropy per node
        """
        bs = [] # collect some partitions
        sum_entropy = 0

        def collect_partitions(s):
            bs.append(s.b.a.copy())

        current_entropy = self.state.entropy(partition_dl=False) # must specify manually

        interval = num_iter // 10

        for i in tqdm(range(0, num_iter)):
                dS, nattempts, nmoves = self.state.mcmc_sweep(niter=1, d=0.00, entropy_args=self.entropy_args)
                current_entropy += dS
                sum_entropy += current_entropy

                collect_partitions(self.state)
                if verbose and i % interval == 0:
                    print("i: {}, dS: {}, nattempts: {}, nmoves: {}".format(i, dS, nattempts, nmoves))

        # Disambiguate partitions and obtain marginals
        pmode = PartitionModeState(bs, converge=True, relabel=True)
        pv = pmode.get_marginal(self.G)

        # Now the node marginals are stored in property map pv. We can
        # visualize them as pie charts on the nodes:
        self.vertex_block_counts = pv
        self.B_max = pmode.get_B()
        
        #calc av entropy
        num_entities = self.G.num_vertices() + self.G.num_edges()
        av_entropy_per_entity = sum_entropy / (num_iter * num_entities)
        if verbose:
            print("Average per node entropy: " + str(av_entropy_per_entity))
            
        return av_entropy_per_entity


    # training methods
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

        return posterior_probs

    
    def generate_feature_matrix(self):
        """
        return X: (N x D) matrix of node features
        X[n, d] = feature d of vertex n
        """
        properties = self.get_feature_names()
        D = len(properties)

        vertices = self.G.get_vertices()
        
        N = len(vertices)
        X = np.empty((N, D))

        for prop_index, prop_name in enumerate(properties):
            value_map = self.get_property_map(prop_name)
            for vertex_index, vertex_id in enumerate(vertices):
                X[vertex_index, prop_index] = value_map[vertex_id]
        
        return X

    
    def sample_classifier_sgld(self, num_iter, step_scaling=1, sigma=1, verbose=False):
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

            return classifier


    def sample_classifier_mala(self, num_iter, step_scaling=1, sigma=1, verbose=False):
        if self.state is None:
            print("No state partition detected >> ABORT")
        else:
            X = self.generate_feature_matrix()
            Y = self.generate_posterior()

            D = X.shape[1]
            B = Y.shape[1]

            classifier = SoftmaxNeuralNet(layers_size=[D, B], sigma=sigma)
            classifier.perform_mala(X, Y, step_scaling=step_scaling, num_iter=num_iter, verbose=verbose)

            return classifier
    

    def sample_classifier_mcmc(self, num_iter, verbose=False):
        if self.state is None:
            print("No state partition detected >> ABORT")
        else:
            properties = self.get_feature_names()
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


    def gen_training_set(self, fraction):
        all_vertices = self.G.get_vertices()
        vertex_list = [vertex for vertex in all_vertices]

        training_set_size = int(len(vertex_list) * fraction)
        training_vertices = np.random.choice(vertex_list, size=training_set_size, replace=False)

        self.training_vertices = training_vertices
        
        test_vertices = vertex_list.copy()
        for vertex in self.training_vertices:
            test_vertices.remove(vertex)

        self.test_vertices = test_vertices

    
    # visualisation
    def draw(self, output=None, gen_layout=True, size=5):
        output = gen_output_path(output)
        vprops = {"size": size}
        
        pos = None
        if gen_layout == False and "_pos" in self.G.vertex_properties:
            pos = self.G.vertex_properties["_pos"]

        if self.state is not None:
            if self.vertex_block_counts is not None:
                print("Drawing soft partition")
                self.state.draw(pos=pos, vertex_shape="pie", vprops=vprops, vertex_pie_fractions=self.vertex_block_counts, output=output)
            else:
                print("Drawing hard state partition")
                self.state.draw(pos=pos, vprops=vprops, output=output)
        else:
            print("No state partition detected >> draw default graph")
            graph_draw(self.G, pos=pos, vprops=vprops, output=output)

    
    def plot_matrix(self):
        if self.state is not None:
            print("Drawing block adjacency matrix $e_{rs}$")
            block_adjacency_matrix = self.state.get_matrix()

            fig = plt.figure()
            ax = plt.subplot(111)

            mat = ax.matshow(block_adjacency_matrix.todense())
            ax.set_title("Block Adjacency Matrix $e_{rs}$")
            ax.set_ylabel("Block index $r$")
            ax.set_xlabel("Block index $s$")
            ax.xaxis.set_label_position("top")
            fig.colorbar(mat)
            
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
            properties = self.get_feature_names()
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

            for prop_name in properties:
                value_map = self.get_property_map(prop_name)
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
            B = len(block_counts)
            x = np.arange(0, B, 1)
            
            plt.bar(x, block_counts)
            plt.title("Block Membership Counts")
            plt.ylabel("Cumulative probability sum")
            plt.xlabel("Block index")

            block_names = [str(num) for num in range(0, B)]
            plt.xticks(ticks=x, labels=block_names)
            plt.show()
        else:
            raise Exception("Vertex block counts not initialised")
