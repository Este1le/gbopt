import unittest
import numpy as np

from test.dummy_graph import graph1d

from gbopt.label_propagation_algorithms.slp import SLP
from gbopt.label_propagation_algorithms.hmn import HMN

class TestLabelPropagationAlgorithms1D(unittest.TestCase):

    def setUp(self):
        self.trueY1d = graph1d.Y
        lids1d = np.random.choice(graph1d.X.shape[0], 10, replace=False)
        Yl1d = graph1d.Y[lids1d]
        graph1d.Y = np.zeros((graph1d.Y.shape[0]))
        graph1d.update(lids1d, Yl1d)
        self.graph1d = graph1d

    def test_hmn_1d(self):
        hmn = HMN(self.graph1d)
        YL = self.graph1d.Y[self.graph1d.lids]
        hmn.label_propagate(self.graph1d)

        assert np.array_equal(self.graph1d.Y[self.graph1d.lids], YL)
        #print("True 1d: ", self.trueY1d[:20])
        #print("Label propagation: ", self.graph1d.Y[:20])
        print("hmn mse", sum((self.trueY1d - self.graph1d.Y)**2))

if __name__ == "__main__":
    unittest.main()
