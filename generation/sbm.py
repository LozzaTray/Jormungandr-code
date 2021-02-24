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

        # must translate from W to e_rs
        edges_between_blocks = np.zeros_like(W)

        for r in range(0, B):
            for s in range(r, B):
                nr = num_vertices_by_block[r]
                ns = num_vertices_by_block[s]
                num_edges_possible = nr * ns

                if r == s:
                    num_edges_possible = nr * (nr-1) / 2
                
                num_edges_expected = num_edges_possible * W[r, s]
                edges_between_blocks[r, s] = num_edges_expected
                edges_between_blocks[r, s] = num_edges_expected

        self.graph = generate_sbm(vertex_block_labels, edges_between_blocks)

    
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
        if self.state is not None:
            print("Drawing state partition")
            self.state.draw(output=output)
        else:
            print("No state partition detected >> draw default graph")
            graph_draw(self.graph, vertex_text=self.graph.vertex_index, output=output)


if __name__ == "__main__":
    n = 1000
    B = 5
    prior = [0.1, 0.15, 0.2, 0.25, 0.3]
    W = (9 * np.identity(B) + 1 * np.ones((B, B))) / 100

    sbm = SBM(n, B, prior, W)
    sbm.partition()
    sbm.draw()