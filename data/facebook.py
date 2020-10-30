import os
import csv


curr_dir = os.path.dirname(__file__)
facebook_dir = os.path.join(curr_dir, "facebook")


class FacebookGraph:
    """
    class for reading and manipulating facebook graphs
    """
    

    def __init__(self, graphId):
        self.graphId = str(graphId)
        self.edges = self.read_edges()
        self.features = self.read_feature_names()
        self.node_features = self.read_node_features()

    
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
        return {feature[0]: feature[1] for feature in features_arr}

    
    def read_node_features(self):
        extension = ".feat"
        delimiter = " "
        node_transform = lambda feature_row: (node_row[0], node_row[1:])
        features_by_node = self.read_graph_file(extension, delimiter, node_transform)
        return {node_feature[0]: node_feature[1] for node_feature in features_by_node}