import os
import csv
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test


curr_dir = os.path.dirname(__file__)
facebook_dir = os.path.join(curr_dir, "facebook")


class FacebookGraph:
    """
    class for reading and manipulating facebook graphs
    """
    

    def __init__(self, graphId, significance_level=99):
        self.graphId = str(graphId)
        self.edges = self.read_edges()
        self.feature_names = self.read_feature_names()
        self.node_features = self.read_node_features()
        self.significance_level = significance_level

    
    def read_graph_file(self, extension, delimiter, transform_function):
        filepath = os.path.join(facebook_dir, self.graphId + extension)
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

    
    def read_node_features(self):
        extension = ".feat"
        delimiter = " "
        node_transform = lambda node_row: (node_row[0], node_row[1:])
        boolean_arr_to_set = lambda boolean_arr: {idx for idx, feature_set in enumerate(boolean_arr) if feature_set == "1"}

        features_by_node = self.read_graph_file(extension, delimiter, node_transform)
        return {node_feature[0]: boolean_arr_to_set(node_feature[1]) for node_feature in features_by_node}


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