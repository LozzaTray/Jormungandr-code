from model.graph_mcmc import Graph_MCMC


def run():
    dataset = "law_firm"

    graph = Graph_MCMC()
    #graph.read_from_gt("cond-mat")
    graph.read_from_ns(dataset)
    graph.remove_property("_pos")
    graph.convert_props_to_flags()
    #graph.filter_edges("layer", 3) # 1: coworkers, 2: friendship, 3: advice

    graph.partition(B_min=2, B_max=10)
    graph.mcmc(1000)
    graph.draw(dataset + ".png")

    classifier = graph.sample_classifier_marginals(2500, step_scaling=0.01)

    names = graph.get_feature_names()

    classifier.sgld_sample_thinning()
    classifier.plot_sampled_weights(names, std_dev_multiplier=2)
    classifier.plot_sample_histogram()
    classifier.plot_sample_history()


if __name__ == "__main__":
    print("Analysing chosen dataset")
    run()