from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test


def run():
    graph = FacebookGraph(3437)
    graph.hypothesis_test_keyword("gender")
    #graph.hypothesis_test_keyword("birthday")
    #graph.hypothesis_test_keyword("first_name")
    #graph.hypothesis_test_keyword("last_name")
    #graph.hypothesis_test_keyword("hometown")
    #graph.hypothesis_test_keyword("language") # not disjoint


if __name__ == "__main__":
    run()