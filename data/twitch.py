import os
import csv
import json
from model.graph_mcmc import Graph_MCMC


curr_dir = os.path.dirname(__file__)
twitch_dir = os.path.join(curr_dir, "twitch")


class TwitchGraph:
    """
    class for reading and manipulating twitch graphs
    """
    

    def __init__(self, country="ENGB"):
        self.country = country
        self.edge_filename = "musae_" + country + "_edges.csv"
        self.feature_filename = "musae_" + country + "_features.json"

        self.edges = self.read_edges()
        self.node_features = self.read_node_features()

    def read_edges(self):
        delimiter = ","
        edge_transform = lambda edge: (edge[0], edge[1])
        filepath = os.path.join(twitch_dir, self.country, self.edge_filename)
        
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            reader.__next__() # discard header
            rows = [edge_transform(row) for row in reader]
        return rows

    def read_node_features(self):
        filepath = os.path.join(twitch_dir, self.country, self.feature_filename)

        with open(filepath) as jsonfile:
            features = json.load(jsonfile)

        features = {key: val for key, val in features.items()}
        return features            

    
    def generate_mcmc_graph(self):
        graph = Graph_MCMC()
        graph.read_from_edges(self.edges)

        feature_set = set()

        for features in self.node_features.values():
            feature_set.union(features)

        vertices = graph.get_vertex_list()

        for feature_name in feature_set:
            value_arr = [feature_name in self.node_features[vertex] for vertex in vertices]
            graph.add_property(feature_name, "bool", value_arr)

        return graph
        



