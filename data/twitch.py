import os
import csv
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.colors import color_between
import pandas as pd
from sklearn import linear_model
from utils.plotting import sorted_bar_plot


curr_dir = os.path.dirname(__file__)
twitch_dir = os.path.join(curr_dir, "twitch")


class TwitchGraph:
    """
    class for reading and manipulating twitch graphs
    """
    

    def __init__(self, country="ENGB"):
        self.country = country
        self.edge_filename = "musae_" + country + "_edges.csv"
        self.feature_filename = "musae_" + country

        self.edges = self.read_edges()
        self.feature_names = self.read_feature_names()
        self.node_features = self.read_node_features()

    
    def read_graph_file(self, filename, delimiter, transform_function):
        filepath = os.path.join(twitch_dir, self.country, filename)
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            rows = [transform_function(row) for row in reader]
        return rows


    def read_edges(self):
        extension = ".edges"
        delimiter = ","
        edge_transform = lambda edge: (edge[0], edge[1])
        filepath = os.path.join(twitch_dir, self.country, self.edge_filename)
        
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            reader.__next__() # discard header
            rows = [transform_function(row) for row in reader]
        return rows

    def read_feature_names(self):
        extension = ".featnames"
        delimiter = " "
        feature_transform = lambda feature_row: (feature_row[0], self.convert_feature_name("_".join(feature_row[1:])))
        features_arr = self.read_graph_file(extension, delimiter, feature_transform)
        return {int(feature[0]): feature[1] for feature in features_arr}

    
    def read_node_features(self):
        extension = ".feat"
        delimiter = " "
        node_transform = lambda node_row: (node_row[0], node_row[1:])
        boolean_arr_to_set = lambda boolean_arr: {idx for idx, feature_set in enumerate(boolean_arr) if feature_set == "1"}

        filename = os.path.join(twitch_dir, self.country, self.feature_filename)

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

    
    def node_has_feature(self, node_id, feature_id):
        node_feature_list = self.node_features.get(node_id)
        return feature_id in node_feature_list


    def feature_name(self, feature_id):
        return self.feature_names.get(feature_id)


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


    def get_feat_ids_and_names(self, keywords=[], bias=True):
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
        if bias == True:
            feat_names_of_interest.append("bias")
        feat_names_of_interest = np.array(feat_names_of_interest)
        return feat_ids_of_interest, feat_names_of_interest


    def gradient_ascent(self, posterior_probs_dict, keywords=[]):
        print("Performing gradient ascent...")

        feat_ids_of_interest, feat_names_of_interest = self.get_feat_ids_and_names(keywords)

        D = len(feat_ids_of_interest) + 1
        X = -1 * np.ones((D, self.N)) # initialise as all -1 and only turn +Ve set features

        node_id_arr = np.zeros(self.N)
        t = np.empty((self.N, 1))

        node_index = 0
        # build X matrix and T array
        for node_id, features in self.node_features.items():
            node_id_arr[node_index] = node_id
            prob = posterior_probs_dict[node_id]
            t[node_index, 0] = 1 if prob > 0.5 else 0

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

        predictions = [1 if prob > 0.5 else 0 for prob in sXw[:, 0]]
        self.plot_space(t, feat_names_of_interest, X)
        self.plot_coeffs(w[:, 0], feat_names_of_interest)
        return w


    def linear_regression(self, output_dict, keywords=[]):
        print("Performing linear regression...")

        feat_ids_of_interest, feat_names_of_interest = self.get_feat_ids_and_names(keywords)

        D = len(feat_names_of_interest)
        X = np.zeros((D, self.N)) # initialise as all -1 and only turn +Ve set features
        X[-1, :] = np.ones(self.N)

        node_id_arr = np.zeros(self.N)
        t = np.empty((self.N, 1))

        node_index = 0
        # build X matrix and t array
        for node_id, features in self.node_features.items():
            node_id_arr[node_index] = node_id
            output = output_dict[node_id]
            t[node_index, 0] = output

            for feat_index, feature_id in enumerate(feat_ids_of_interest):
                if feature_id in features:
                    X[feat_index, node_index] = 1

            node_index += 1

        Xt = np.transpose(X)
        regr = linear_model.LinearRegression(fit_intercept=False) # X already has constant term for bias
        regr.fit(Xt, t)
        r_squared = regr.score(Xt, t)
        print("R^2 = {}".format(r_squared))
        self.plot_features(X, feat_names_of_interest, t)
        self.plot_coeffs(regr.coef_[0], feat_names_of_interest)


    def plot_features(self, X, feat_names, t):
        feature_totals = np.sum(X, axis=1)
        feature_totals_in_comunity_one = np.zeros_like(feature_totals)
        for i in range(0, self.N):
            if t[i, 0] > 0:
                feature_totals_in_comunity_one[:] += X[:, i]

        fraction_in_community_one = np.divide(feature_totals_in_comunity_one, feature_totals)
        sorted_bar_plot(fraction_in_community_one, feat_names, "Community fractions", "Fraction of feature total in S")


    def plot_coeffs(self, weights, feat_names):
        sorted_bar_plot(weights, feat_names, "Regression of $\\tilde{\sigma}_v$ on select features", "Regression coefficient")


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


    def convert_feature_name(self, feature_name):
        pretty_name = feature_name.replace(";", "-")
        pretty_name = pretty_name.replace("anonymized_feature_", "")
        return pretty_name

