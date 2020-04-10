import unittest
import numpy as np

from gbopt.label_propagation_algorithms.hmn import HMN
from gbopt.acquisition_functions.eif import EIF
from gbopt.maximizers.random_sampling import RandomSampling
from gbopt.solver.gbopt import GBOPT
from gbopt.initial_design.init_random_choice import init_random_choice

from test.dummy_graph import graph1d
from test.dummy_graph import graph2d
from test.dummy_graph import objective_function1d
from test.dummy_graph import objective_function2d

class TestGbopt(unittest.TestCase):

    def setUp(self):
        self.true1dY = graph1d.Y
        self.true2dY = graph2d.Y
        graph1d.Y = np.zeros((graph1d.Y.shape[0])).reshape(graph1d.Y.shape[0], -1)
        graph2d.Y = np.zeros((graph2d.Y.shape[0])).reshape(graph2d.Y.shape[0], -1)
        lp1 = HMN(graph1d)
        lp2 = HMN(graph2d)
        af1d = EIF(graph1d, lp1)
        af2d = EIF(graph2d, lp2)
        maximizer1d = RandomSampling(af1d)
        maximizer2d = RandomSampling(af2d)
        self.graph1d = graph1d
        self.graph2d = graph2d


        self.solver1d = GBOPT(self.graph1d, objective_function1d, lp1,
                              af1d, maximizer1d,
                              init_random_choice)

        self.solver2d = GBOPT(self.graph2d, objective_function2d, lp2,
                              af2d, maximizer2d,
                              init_random_choice)

    def test_run_1d(self):
        n_iters = 10
        inc, inc_val = self.solver1d.run(num_iterations = n_iters)
        print("inc: ", inc, "inc_val: ", inc_val)
        print("preds for one point: ", np.array(self.solver1d.predictions)[:,5])
        mse = [np.sum(np.square(p-self.true1dY)) for p in self.solver1d.predictions]
        print("mse: ", mse)
        assert len(inc) == 1
        assert len(inc_val) == 1
        assert np.array(inc) >= 0
        assert np.array(inc) <= 10
        n_iters = min(n_iters, self.solver1d.graph.X.shape[0])
        assert len(self.solver1d.incumbents_values) == n_iters
        assert len(self.solver1d.incumbents) == n_iters
        assert len(self.solver1d.time_overhead) == n_iters
        assert len(self.solver1d.time_func_evals) == n_iters
        assert len(self.solver1d.runtime) == n_iters
        assert self.solver1d.graph.lids.shape[0] == n_iters

    # def test_run_2d(self):
    #     n_iters = 10
    #     inc, inc_val = self.solver2d.run(n_iters)
    #     print("inc: ", inc, "inc_val: ", inc_val)
    #     print("preds for one point: ", np.array(self.solver2d.predictions)[:,5])
    #     mse = [np.sum(np.square(p-self.true2dY)) for p in self.solver2d.predictions]
    #     print("mse: ", mse)
    #     assert len(inc) == 2
    #     assert len(inc_val) == 1
    #     assert np.all(np.array(inc) >= 0)
    #     assert np.all(np.array(inc) <= 10)
    #     n_iters = min(n_iters, self.solver2d.graph.X.shape[0])
    #     assert len(self.solver2d.incumbents_values) == n_iters
    #     assert len(self.solver2d.incumbents) == n_iters
    #     assert len(self.solver2d.time_overhead) == n_iters
    #     assert len(self.solver2d.time_func_evals) == n_iters
    #     assert len(self.solver2d.runtime) == n_iters
    #     assert self.solver2d.graph.lids.shape[0] == n_iters

    # def test_choose_next_1d(self):
    #     self.graph1d.lids = None
    #     id_new = self.solver1d.choose_next(self.graph1d)
    #     print("In test,", type(id_new))
    #     assert type(id_new) == int
    #     assert id_new >= 0
    #     assert id_new <= 100
    #
    # def test_choose_next_2d(self):
    #     self.graph2d.lids = None
    #     id_new = self.solver2d.choose_next(self.graph2d)
    #     assert type(id_new) == int
    #     assert id_new >= 0
    #     assert id_new <= 100

if __name__ == "__main__":
    unittest.main()
