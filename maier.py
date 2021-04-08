from data.polblogs import PolBlog
from model.graph_mcmc import Graph_MCMC


def run():
    graph = Graph_MCMC()
    #graph.read_from_gt("cond-mat")
    graph.read_from_ns("facebook_friends")
    graph.remove_property("_graphml_vertex_id")
    graph.remove_property("_pos")
    graph.convert_props_to_flags()
    graph.add_ego_node()

    graph.partition(B_min=4, B_max=20)
    graph.mcmc(1000, verbose=False)
    graph.draw("maier-fb.png")

    classifier = graph.sample_classifier_marginals(2500, step_scaling=0.01, verbose=False)

    names = graph.get_feature_names()

    classifier.sgld_sample_thinning()
    classifier.plot_sampled_weights(names, std_dev_multiplier=2)
    classifier.plot_sample_histogram()
    classifier.plot_sample_history()

if __name__ == "__main__":
    print("Analysing Maier facebook friends")
    run()