import unittest
import numpy as np

from gbopt.maximizers.random_sampling import RandomSampling
from gbopt.acquisition_functions.base_acquisition import BaseAcquisitionFunction
from gbopt.label_propagation_algorithms.hmn import HMN
from test.dummy_graph import graph1d
from test.dummy_graph import graph2d

class DemoAcquisitionFunction1d(BaseAcquisitionFunction):

    def __init__(self):
        self.graph1d = graph1d
        lp = HMN(self.graph1d)
        super(DemoAcquisitionFunction1d, self).__init__(self.graph1d, lp)

    def compute(self, i):
        y = (5 - self.graph1d.X[i]) ** 2
        return y

class DemoAcquisitionFunction2d(BaseAcquisitionFunction):

    def __init__(self):
        self.graph2d = graph2d
        lp = HMN(self.graph2d)
        super(DemoAcquisitionFunction2d, self).__init__(self.graph2d, lp)

    def compute(self, i):
        y = (5 - self.graph2d.X[i]) ** 2
        return y

class TestMaximizers(unittest.TestCase):

    def setUp(self):
        self.graph1d = graph1d
        self.graph2d = graph2d
        self.graph1d.lids = np.random.choice(graph1d.X.shape[0], 10, replace=False)
        self.graph2d.lids = np.random.choice(graph2d.X.shape[0], 10, replace=False)
        self.objective_function1d = DemoAcquisitionFunction1d()
        self.objective_function2d = DemoAcquisitionFunction2d()

    def test_random_sampling_1d(self):
        maximizer = RandomSampling(self.objective_function1d)
        id1 = maximizer.maximize(self.graph1d)

        assert np.array([id1]).shape[0] == 1
        assert id1 >= 0
        assert id1 <= 100

    def test_random_sampling_2d(self):
        maximizer = RandomSampling(self.objective_function2d)
        id2 = maximizer.maximize(self.graph2d)

        assert np.array([id2]).shape[0] == 1
        assert id2 >= 0
        assert id2 <= 120

if __name__ == "__main__":
    unittest.main()



