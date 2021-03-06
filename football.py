from model.graph_mcmc import Graph_MCMC


def run():
    #dataset = "escorts"
    dataset = "football_tsevans"

    graph = Graph_MCMC()
    #graph.read_from_gt("cond-mat")
    graph.read_from_ns(dataset)
    graph.remove_property("_graphml_vertex_id")
    graph.remove_property("_pos")
    graph.remove_property("label")
    #graph.convert_props_to_flags()
    graph.convert_to_flags("value", "conf-")

    graph.partition(B_min=4, B_max=20)
    graph.mcmc(1000)
    graph.draw(dataset + ".png")

    classifier = graph.sample_classifier_marginals(2500, step_scaling=0.01)

    names = graph.get_feature_names()

    classifier.thin_samples()
    classifier.plot_sampled_weights(names, std_dev_multiplier=2)
    classifier.plot_sample_histogram()
    classifier.plot_sample_history()


if __name__ == "__main__":
    print("Analysing chosen dataset")
    run()