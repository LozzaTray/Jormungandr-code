from data.twitch import TwitchGraph

def run():
    print("Loading data...")
    twitch = TwitchGraph()
    print("Converting to graph-tool...")
    graph = twitch.generate_mcmc_graph()
    print("Done")

    graph.partition(B_min=4, B_max=20)
    graph.mcmc(1000, verbose=True)
    graph.draw("twitch.png")

    graph.plot_posterior_props()
    names = graph.get_feature_names()


    # classifier = graph.sample_classifier_marginals(2500, step_scaling=0.001, verbose=True)
    classifier = graph.sample_classifier_mala(2500, step_scaling=0.001, verbose=True)
    classifier.thin_samples()


    #classifier.plot_U()
    classifier.plot_block_principal_dims(names.copy(), 1)
    classifier.plot_sampled_weights(names.copy(), std_dev_multiplier=1, null_space=1)


if __name__ == "__main__":
    run()