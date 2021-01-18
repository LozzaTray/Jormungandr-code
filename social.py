from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test
from model.graph import Graph


def run():
    ego_id = 0 # 0 or 107
    gender_index = 77 # 77 or 264
    fb = FacebookGraph(ego_id)
    fb.hypothesis_test_threeway(gender_index)
    #fb.hypothesis_test_keyword("gender")

    nodes_by_gender = fb.get_node_has_feature_dict(gender_index)
    graph = Graph(fb.edges)
    #community_probs = graph.abp()
    #graph.proportions_in_each(nodes_by_gender)
    #graph.draw_partition(nodes_by_gender)
    graph.draw_standard(nodes_by_gender, "Egonet id-{}".format(ego_id), "gender-77")
    #graph.draw_standard()
    #fb.gradient_ascent(community_probs)
    #fb.linear_regression(community_probs, ["language", "gender"])
    #fb.linear_regression(community_probs)


if __name__ == "__main__":
    print("Analysing facebook data")
    run()