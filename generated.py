from model.graph_mcmc import Graph_MCMC


def run():
    graph = Graph_MCMC([])
    graph.read_from_file("generated.gml")

    graph.partition(B_min=2, B_max=5)

    #ml_classifier = graph.train_feature_classifier()
    #ml_classifier.plot_final_weights(["a", "b", "c"])

    classifier = graph.sample_classifier_mcmc(100, True)
    classifier.plot_sampled_weights(["a", "b", "c"])



if __name__ == "__main__":
    print("Analysing simulated data")
    run()