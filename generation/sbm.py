from graph_tool import Graph
from graph_tool.generation import generate_maxent_sbm
from graph_tool.all import graph_draw
from graph_tool.inference import minimize_blockmodel_dl
import numpy as np


class SBM:

    def __init__(self, n, B, prior, W):
        """
        Draw new SBM with:
            n - vertices
            prior - B length array denoting fraction in each community
            W - BxB connectivity matrix
        """

        assert B == len(prior)
        num_vertices_by_block = np.random.multinomial(n, prior)

        vertex_block_labels = []
        for idx, num_in_block in enumerate(num_vertices_by_block):
            new_block_labels = [idx] * num_in_block # repeats num_in_block times [idx, idx .... idx]
            vertex_block_labels.extend(new_block_labels)

        out_fugacities = [1] * n

        self.n = n
        self.B = B
        self.num_vertices_by_block = num_vertices_by_block
        self.true_block_labels = vertex_block_labels
        self.graph = generate_maxent_sbm(vertex_block_labels, W, out_fugacities)

    
    def add_property(self, name, prob_by_block):
        """
        Add property to node
            name: string - property name
            prob_by_block: array - B length array with probability feature turned on for each block
        """

        value_sequence = []

        for b in range(0, self.B):
            num_vertices = self.num_vertices_by_block[b]
            prob = prob_by_block[b]
            num_successes = np.random.binomial(num_vertices, prob)
            value_sequence.extend([1] * num_successes)
            value_sequence.extend([0] * (num_vertices - num_successes))

        vertex_prop = self.graph.new_vertex_property("int", value_sequence)
        self.graph.vertex_properties[name] = vertex_prop # add to graph

    
    def partition(self, B_min=None, B_max=None, degree_corrected=True):
        """
        Performs MCMC algorithm to minimise description length (DL)
        returns partition array
        """
        print("Performing inference...")
        self.state = minimize_blockmodel_dl(self.graph, B_min=B_min, B_max=B_max, deg_corr=degree_corrected, verbose=True)
        print("Done")
        return self.state.get_blocks()


    def draw(self, output=None):
        output = self.gen_output_path(output)
        if self.state is not None:
            print("Drawing state partition")
            self.state.draw(output=output)
        else:
            print("No state partition detected >> draw default graph")
            graph_draw(self.graph, vertex_text=self.graph.vertex_index, output=output)

    
    def gen_output_path(self, filename):
        if filename is not None:
            return "output/" + filename
        return None

    
    def save(self, output=None):
        """Recommend .gml format"""
        output = self.gen_output_path(output)
        self.graph.save(output)


if __name__ == "__main__":
    n = 1000
    B = 3
    prior = [0.33, 0.33, 0.34]
    W = (9 * np.identity(B) + 1 * np.ones((B, B))) / 200

    sbm = SBM(n, B, prior, W)
    
    sbm.add_property("property-a", [0.8, 0.1, 0.1])
    sbm.add_property("property-b", [0.1, 0.8, 0.1])
    sbm.add_property("property-c", [0.1, 0.1, 0.8])

    sbm.partition(3, 7)
    sbm.draw("generated.png")
    sbm.save("generated.gml")