from data.polblogs import PolBlog
from model.graph_mcmc import Graph_MCMC


def run():
    graph = Graph_MCMC([])
    graph.read_from_file("polblogs.gml")
    graph.filter_out_low_degree(2)
    graph.remove_property("label")
    graph.remove_property("source")
    graph.partition(B_min=1, B_max=2)
    #graph.draw("polblogs.png")
    graph.mcmc(100)
    classifier = graph.sample_classifier_marginals(1000, verbose=True)

    classifier.sgld_sample_thinning()
    classifier.plot_sampled_weights(["Right-Wingness"])
    classifier.plot_sample_histogram()
    classifier.plot_sample_history()

if __name__ == "__main__":
    print("Analysing political blogs")
    run()