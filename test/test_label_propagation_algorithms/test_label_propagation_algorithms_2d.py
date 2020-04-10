import unittest
import numpy as np

from test.dummy_graph import graph2d

from gbopt.label_propagation_algorithms.slp import SLP
from gbopt.label_propagation_algorithms.hmn import HMN

class TestLabelPropagationAlgorithms2D(unittest.TestCase):

    def setUp(self):
        self.trueY2d = graph2d.Y
        lids2d = np.random.choice(graph2d.X.shape[0], 10, replace=False)
        Yl2d = graph2d.Y[lids2d]
        graph2d.Y = np.zeros((graph2d.Y.shape[0]))
        graph2d.update(lids2d, Yl2d)
        self.graph2d = graph2d

    def test_hmn_2d(self):
        hmn = HMN(self.graph2d)
        YL = self.graph2d.Y[self.graph2d.lids]
        hmn.label_propagate(self.graph2d)

        assert np.array_equal(self.graph2d.Y[self.graph2d.lids], YL)
        #print("True 2d: ", self.trueY2d[:20])
        #print("Label propagation: ", self.graph2d.Y[:20])
        print("hmn mse", sum((self.trueY2d - self.graph2d.Y)**2))

if __name__ == "__main__":
    unittest.main()
