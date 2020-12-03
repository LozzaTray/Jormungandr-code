import os
import csv
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test
from distributions.sigmoid import sigmoid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.colors import color_between

curr_dir = os.path.dirname(__file__)
facebook_dir = os.path.join(curr_dir, "facebook")


class FacebookGraph:
    """
    class for reading and manipulating facebook graphs
    """
    

    def __init__(self, ego_id, significance_level=99):
        self.ego_id = str(ego_id)
        self.edges = self.read_edges()
        self.feature_names = self.read_feature_names()
        self.node_features = self.read_node_features()
        self.significance_level = significance_level

        # add edges from ego node
        for node_id in self.node_features.keys():
            self.edges.append((self.ego_id, node_id))

        # add ego features
        self.node_features[self.ego_id] = self.read_ego_features()

        # constants
        self.N = len(self.node_features)
        self.M = len(self.edges)
        self.D = len(self.feature_names)

    
    def read_graph_file(self, extension, delimiter, transform_function):
        filepath = os.path.join(facebook_dir, self.ego_id + extension)
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            rows = [transform_function(row) for row in reader]
        return rows


    def read_edges(self):
        extension = ".edges"
        delimiter = " "
        edge_transform = lambda edge: (edge[0], edge[1])
        return self.read_graph_file(extension, delimiter, edge_transform)


    def read_feature_names(self):
        extension = ".featnames"
        delimiter = " "
        feature_transform = lambda feature_row: (feature_row[0], "_".join(feature_row[1:]))
        features_arr = self.read_graph_file(extension, delimiter, feature_transform)
        return {int(feature[0]): feature[1] for feature in features_arr}


    def read_ego_features(self):
        extension = ".egofeat"
        delimiter = " "

        boolean_arr_to_set = lambda boolean_arr: {idx for idx, feature_set in enumerate(boolean_arr) if feature_set == "1"}
        feature_set = self.read_graph_file(extension, delimiter, boolean_arr_to_set)
        return feature_set[0]

    
    def read_node_features(self):
        extension = ".feat"
        delimiter = " "
        node_transform = lambda node_row: (node_row[0], node_row[1:])
        boolean_arr_to_set = lambda boolean_arr: {idx for idx, feature_set in enumerate(boolean_arr) if feature_set == "1"}

        features_by_node_arr = self.read_graph_file(extension, delimiter, node_transform)
        return {node_feature[0]: boolean_arr_to_set(node_feature[1]) for node_feature in features_by_node_arr}


    def same_community(self, node_a, node_b, feature_id):
        node_a_features = self.node_features.get(node_a)
        node_b_features = self.node_features.get(node_b)
        return (feature_id in node_a_features) == (feature_id in node_b_features)


    def same_community_multiple(self, node_a, node_b, feature_ids):
        node_a_features = self.node_features.get(node_a)
        node_b_features = self.node_features.get(node_b)
        for feature_id in feature_ids:
            if (feature_id in node_a_features) and (feature_id in node_b_features):
                return True
        return False


    def hypothesis_test_single(self, feature_id):
        """Is there evidence to suggest this feature affects how people interact"""
        print("Testing whether feature-{}: {}".format(feature_id, self.feature_names.get(feature_id)))
        print("Impacts probability of two random individuals being FB friends\n")

        N = len(self.node_features) # num nodes in graph
        N_1 = 0 # num nodes with given feature_id
        for features in self.node_features.values():
            if feature_id in features:
                N_1 += 1

        N_2 = N - N_1 # num nodes without given feature
        print("Num nodes with/without feature:\nN_1 = {} , N_2 = {}\n".format(N_1, N_2))

        E_max = int(N * (N + 1) / 2) # max edges possible

        m = N_1 * N_2 # num edges possible between communities
        n = E_max - m # num edges possible within communities

        k = 0 # num edges within same community
        l = 0 # num edges between communities

        for edge in self.edges:
            if self.same_community(edge[0], edge[1], feature_id):
                k += 1
            else:
                l += 1

        t, p = two_samples_mean_ll_ratio(n, m, k, l)
        print("t-statistic: t = {:.3f}".format(t))
        print("p-value: p = {:.5f}\n".format(p))

        if (100*p < (100 - self.significance_level)):
            print("Null Hypothesis rejected at the {}% significance level".format(self.significance_level))
            print("Feature impacts friendship probability")
        else:
            print("Insufficient evidence to reject null")


    def hypothesis_test_threeway(self, feature_id):
        print("Testing whether feature-{}: {}".format(feature_id, self.feature_names.get(feature_id)))
        print("Impacts probability of two random individuals being FB friends\n")

        N = len(self.node_features) # num nodes in graph
        N_a = 0 # num nodes with given feature_id
        for features in self.node_features.values():
            if feature_id in features:
                N_a += 1

        N_b = N - N_a # num nodes without given feature
        print("Num nodes with/without feature:\nN_a = {} , N_b = {}\n".format(N_a, N_b))

        a2a = 0
        a2b = 0
        b2b = 0


        for edge in self.edges:
            origin_is_A = feature_id in self.node_features.get(edge[0])
            dest_is_A = feature_id in self.node_features.get(edge[1])

            if origin_is_A and dest_is_A:
                a2a += 1
            elif origin_is_A or dest_is_A:
                a2b += 1
            else:
                b2b += 1

        a2a_max = int(N_a * (N_a + 1) / 2)
        a2b_max = N_a * N_b
        b2b_max = int(N_b * (N_b + 1) / 2)

        print("Link proportions:\na2a={:.3e} , a2b={:.3e} , b2b={:.3e}\n".format(a2a / a2a_max, a2b / a2b_max, b2b / b2b_max))

        t, p1 = two_samples_mean_ll_ratio(a2a_max, a2b_max, a2a, a2b)
        t, p2 = two_samples_mean_ll_ratio(a2a_max, b2b_max, a2a, b2b)
        t, p3 = two_samples_mean_ll_ratio(a2b_max, b2b_max, a2b, b2b)

        print("p-values for (1: a2a v a2b ; 2: a2a v b2b ; 3: a2b v b2b):")
        print("p1 = {:.3e} , p2 = {:.3e} , p3 = {:.3e}".format(p1, p2, p3))


    def hypothesis_test_multi_group(self, feature_ids):
        """Is there evidence to suggest this feature affects how people interact"""
        print("Testing whether features: {}".format(feature_ids))
        print("values: {}".format([self.feature_names.get(feature_id) for feature_id in feature_ids]))
        print("Impact probability of two random individuals being FB friends\n")

        num_communities = len(feature_ids)
        N_arr = [0] * num_communities # length
        nodes = set()

        for i in range(0, num_communities):
            feature_id = feature_ids[i]
            for node, features in self.node_features.items():
                if feature_id in features:
                    N_arr[i] += 1
                    nodes.add(node)

        N = len(nodes)
        if N != sum(N_arr):
            raise ValueError("Sets not disjoint")

        print("Num nodes in each category:\n{}\n".format(N_arr))
        E_max = int(N * (N + 1) / 2) # max edges possible

        m = 0 # num edges possible between communities
        for i in range(0, num_communities):
            for j in range(i+1, num_communities):
                m += N_arr[i] * N_arr[j]
        
        n = E_max - m # num edges possible within communities

        k = 0 # num edges within same community
        l = 0 # num edges between communities

        for edge in self.edges:
            if edge[0] in nodes and edge[1] in nodes:
                if self.same_community_multiple(edge[0], edge[1], feature_ids):
                    k += 1
                else:
                    l += 1

        t, p = two_samples_mean_ll_ratio(n, m, k, l, debug=True)
        # print("t-statistic: t = {:.3f}".format(t))
        # print("p-value: p = {:.5f}\n".format(p))

        if (100*p < (100 - self.significance_level)):
            print("Null Hypothesis rejected at the {}% significance level".format(self.significance_level))
            print("Feature impacts friendship probability")
        else:
            print("Insufficient evidence to reject null")


    def hypothesis_test_keyword(self, keyword):
        feature_ids = []
        for feature_id, feature_name in self.feature_names.items():
            if keyword in feature_name:
                feature_ids.append(feature_id)
                
        return self.hypothesis_test_multi_group(feature_ids)


    def get_node_set(self):
        origin_vertices = set([edge[0] for edge in self.edges])
        destin_vertices = set([edge[1] for edge in self.edges])
        return origin_vertices.union(destin_vertices)

    
    def get_node_has_feature_dict(self, feature_id):
        node_set = self.get_node_set()
        node_has_feature = {}

        for node in node_set:
            node_features = self.node_features.get(node)
            if feature_id in node_features:
                node_has_feature[node] = True
            else:
                node_has_feature[node] = False

        return node_has_feature


    def gradient_ascent(self, posterior_probs_dict, keywords=[]):
        print("Performing gradient ascent...")

        feat_ids_of_interest = []
        feat_names_of_interest = []
        
        if len(keywords) == 0:
            feat_ids_of_interest.extend(self.feature_names.keys())
            feat_names_of_interest.extend(self.feature_names.values())
        else:
            for feature_id, feature_name in self.feature_names.items():
                for keyword in keywords:
                    if keyword in feature_name:
                        feat_ids_of_interest.append(feature_id)
                        feat_names_of_interest.append(feature_name)
                        break
        feat_names_of_interest.append("bias")
        feat_names_of_interest = np.array(feat_names_of_interest)

        D = len(feat_ids_of_interest) + 1
        X = -1 * np.ones((D, self.N)) # initialise as all -1 and only turn +Ve set features

        node_id_arr = np.zeros(self.N)
        t = np.zeros((self.N, 1))

        node_index = 0
        # build X matrix and T array
        for node_id, features in self.node_features.items():
            node_id_arr[node_index] = node_id
            t[node_index, 0] = posterior_probs_dict[node_id]

            for feat_index, feature_id in enumerate(feat_ids_of_interest):
                if feature_id in features:
                    X[feat_index, node_index] = 1

            node_index += 1

        const = np.matmul(X, t - 1)
        w = np.random.randn(D, 1)

        max_iters = 100
        alpha = 0.1
        eta = 1
        rate = 0.95

        for i in tqdm(range(0, max_iters)):
            Xw = np.matmul(np.transpose(X), w)
            sXw = sigmoid(Xw)
            w = w + eta * (const + np.matmul(X, sXw) - alpha * w)
            eta = eta * rate

        self.plot_space(t, feat_names_of_interest, X)
        self.plot_performance(t - sXw, w, feat_names_of_interest)
        return w


    def plot_performance(self, error, w, feat_names):
        root_mean_squared_error = np.sqrt(np.mean(np.square(error)))
        print("rms error: {}".format(root_mean_squared_error))

        sorted_weight_indices = np.argsort(np.abs(w[:,0]))
        plt.figure()
        plt.barh(feat_names[sorted_weight_indices], w[sorted_weight_indices, 0])
        plt.show()


    def plot_space(self, t, feat_names, X):
        x = X[0, :]
        y = X[1, :]
        z = t[:, 0]

        colors = [color_between(val) for val in z]

        plt.figure()
        plt.scatter(x, y, c=colors)
        plt.xlabel(feat_names[0])
        plt.ylabel(feat_names[1])
        plt.title("Colour is prob in partition 1")
        plt.show()

