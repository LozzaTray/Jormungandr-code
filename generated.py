from model.graph_mcmc import Graph_MCMC


def run():
    graph = Graph_MCMC([])
    graph.read_from_file("generated.gml")

    graph.partition(B_min=2, B_max=5)
    graph.draw("auto-gen.png")
    graph.plot_matrix()
    graph.plot_community_property_fractions()
    graph.train_feature_classifier()


if __name__ == "__main__":
    print("Analysing simulated data")
    run()