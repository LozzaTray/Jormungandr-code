import networkx as nx
import matplotlib.pyplot as plt
import os

curr_dir = os.path.dirname(__file__)
pol_dir = os.path.join(curr_dir, "polblogs")

class PolBlog:

    def __init__(self):
        print("Reading in PolBlog data...")
        filepath = os.path.join(pol_dir, "polblogs.gml")
        self.graph = nx.read_gml(filepath)
        print("Done")


    def draw(self):
        print("Drawing network...")
        nx.draw_networkx(self.graph)
        plt.show()