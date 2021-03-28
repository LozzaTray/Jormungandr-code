from data.polblogs import PolBlog
from model.graph_mcmc import Graph_MCMC


def run():
    graph = Graph_MCMC([])
    graph.read_from_file("polblogs.gml")
    graph.filter_out_low_degree(2)
    graph.partition(B_min=1, B_max=2)
    graph.draw("polblogs.png")

if __name__ == "__main__":
    print("Analysing political blogs")
    run()