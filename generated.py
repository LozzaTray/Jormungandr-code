from model.graph_mcmc import Graph_MCMC

def run():
    graph = Graph_MCMC()
    graph.read_from_file("generated.gml")

    feature_names = graph.get_feature_names()
    graph.partition(B_min=2, B_max=5)

    # ml_classifier = graph.train_map_classifier()
    # ml_classifier.plot_final_weights(feature_names)

    # mcmc_classifier = graph.sample_classifier_mcmc(100, True)
    # mcmc_classifier.plot_sampled_weights(feature_names)

    _B_max = graph.mcmc(10, verbose=True)
    marginal_classifier = graph.sample_classifier_marginals(2500, step_scaling=0.001, sigma=1, verbose=True)
    marginal_classifier.plot_sampled_weights(feature_names)
    marginal_classifier.plot_sample_histogram()
    marginal_classifier.plot_sample_history()



if __name__ == "__main__":
    print("Analysing simulated data")
    run()