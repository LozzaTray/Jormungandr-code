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

    graph.plot_posterior_props()
    names = graph.get_feature_names()

    # map classifier
    # map_classifier = graph.train_map_classifier()
    # map_classifier.plot_final_weights(names)


    # classifier = graph.sample_classifier_marginals(2500, step_scaling=0.001, verbose=True)
    classifier = graph.sample_classifier_mala(2500, step_scaling=0.001, verbose=True)
    classifier.thin_samples()


    #classifier.plot_U()
    #classifier.plot_final_weights(names)
    classifier.plot_block_principal_dims(names.copy(), 1)
    classifier.plot_sampled_weights(names.copy(), std_dev_multiplier=1, null_space=1)
    #classifier.plot_sample_histogram()
    #classifier.plot_sample_history()

if __name__ == "__main__":
    print("Analysing Maier facebook friends")
    run()