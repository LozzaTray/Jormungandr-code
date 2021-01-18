from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test
from model.graph import Graph


def run():
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


if __name__ == "__main__":
    print("Analysing facebook data")
    run()