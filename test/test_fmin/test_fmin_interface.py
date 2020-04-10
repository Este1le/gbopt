import unittest

import numpy as np

from gbopt.fmin.gbopt import gb_opt

from test.dummy_graph import objective_function1d
from test.dummy_graph import objective_function2d
from test.dummy_graph import graph1d
from test.dummy_graph import graph2d

class TestFminInterface(unittest.TestCase):
    def test_gbopt_hmn_1d(self):
        res = gb_opt(graph1d, objective_function1d, label_prop_alg="hmn",
                     n_init=3, num_iterations=10)

        print("x_opt, f_opt:", res["x_opt"], res["f_opt"])
        print("predictions:", res["predictions"][-1][5])
        print("incumbents:", res["incumbents"])
        print("incumbents_values:", res["incumbents_values"])
        print("runtime:", res["runtime"])
        print("overhead:", res["overhead"])
        print("labeled_indices:", res["labeled_indices"])
        print("X:", res["X"][-1])
        print("Y:", res["y"])
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 10

    def test_gbopt_hmn_2d(self):
        res = gb_opt(graph2d, objective_function2d, label_prop_alg="hmn",
                     n_init=3, num_iterations=10)

        print("x_opt, f_opt:", res["x_opt"], res["f_opt"])
        assert len(res["x_opt"]) == 2
        assert np.all(np.array(res["x_opt"]) >= 0)
        assert np.all(np.array(res["x_opt"]) <= 10)

if __name__ == "__main__":
    unittest.main()
