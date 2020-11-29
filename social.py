from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test
from model.graph import Graph


def run():
    num = 0
    fb = FacebookGraph(num)
    #fb.hypothesis_test_threeway(77)
    #fb.hypothesis_test_keyword("gender")
    #fb.hypothesis_test_keyword("birthday")
    #fb.hypothesis_test_keyword("first_name")
    #fb.hypothesis_test_keyword("last_name")
    #fb.hypothesis_test_keyword("hometown")
    #fb.hypothesis_test_keyword("language") # not disjoint
    nodes_by_gender = fb.get_node_has_feature_dict(77)
    graph = Graph(fb.edges)
    graph.abp()
    graph.proportions_in_each(nodes_by_gender)
    graph.draw_partition(nodes_by_gender)
    graph.draw_standard(nodes_by_gender)
    graph.draw_standard()


if __name__ == "__main__":
    print("Analysing facebook data")
    run()