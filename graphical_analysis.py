from data.facebook import FacebookGraph
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test

significance_level = 95

def run():
    graph = FacebookGraph(0)
    graph.hypothesis_test(77)


if __name__ == "__main__":
    run()