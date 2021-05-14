from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test
from model.graph import Graph
from model.graph_mcmc import Graph_MCMC
import numpy as np


def standard():
    ego_id = 0 # 0 or 107
    gender_index = 77 # 77 or 264
    fb = FacebookGraph(ego_id)
    #fb.hypothesis_test_threeway(gender_index)
    #fb.hypothesis_test_keyword("gender")

    nodes_by_gender = fb.get_node_has_feature_dict(gender_index)
    graph = Graph(fb.edges)
    community_output = graph.abp(seed=1)
    graph.proportions_in_each(nodes_by_gender)

    #graph.draw_standard(title="Egonet id-{} post ABP".format(ego_id))
    #graph.draw_partition(nodes_by_gender, "Egonet id-{} grouped by ABP hard partition".format(ego_id), "gender-77")
    #graph.draw_standard(nodes_by_gender, "Egonet id-{}".format(ego_id), "gender-77")
    #fb.gradient_ascent(community_output)
    fb.linear_regression(community_output, ["gender", "language"])
    #fb.linear_regression(community_output)

    
def mcmc():
    ego_id = 0 # 0 or 107
    fb = FacebookGraph(str(ego_id))
    int_edges = [ (int(edge[0]), int(edge[1])) for edge in fb.edges]
    int_edges = np.array(int_edges)

    graph = Graph_MCMC()
    graph.read_from_edges(int_edges)
    vertices = graph.get_vertex_list()
    
    selected_feat_ids, selected_feat_names = fb.get_feat_ids_and_names(keywords=["gender", "language", "hometown", "last_name"])
    #selected_feat_ids, selected_feat_names = fb.get_feat_ids_and_names()


    for feat_id in selected_feat_ids:
        feature_name = fb.feature_name(feat_id)
        feature_flags = [fb.node_has_feature(str(vertex), feat_id) for vertex in vertices]
        graph.add_property(feature_name, "bool", feature_flags)

    graph.partition(B_min=2, B_max=10)
    graph.mcmc(num_iter=100)
    #graph.draw("partition.png")
    #graph.plot_matrix()
    #graph.plot_community_property_fractions()

    marginal_classifier = graph.sample_classifier_marginals(1000, step_scaling=0.01, sigma=1, verbose=True)    
    
    feature_names = graph.get_feature_names()

    marginal_classifier.thin_samples()
    marginal_classifier.plot_sampled_weights(feature_names, std_dev_multiplier=2)
    marginal_classifier.plot_sample_histogram()
    marginal_classifier.plot_sample_history()


if __name__ == "__main__":
    print("Analysing facebook data")
    mcmc()