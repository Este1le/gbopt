import unittest
import numpy as np
from test.dummy_graph import graph1d
from gbopt.acquisition_functions.eif import EIF
from gbopt.label_propagation_algorithms.hmn import HMN

class TestEIF2D(unittest.TestCase):

    def setUp(self):
        self.graph1d = graph1d
        lids1d = np.random.choice(graph1d.X.shape[0], 10, replace=False)
        uids = np.delete(np.arange(self.graph1d.Y.shape[0]), lids1d)
        self.ym = np.argmax(self.graph1d.Y[uids])
        Yl1d = graph1d.Y[lids1d]
        graph1d.Y = np.zeros((graph1d.Y.shape[0]))
        graph1d.update(lids1d, Yl1d)
        self.lp = HMN(self.graph1d)

    def test_compute(self):
        self.lp.label_propagate(self.graph1d)
        eif = EIF(self.graph1d, self.lp)
        uids = np.delete(np.arange(self.graph1d.Y.shape[0]), self.graph1d.lids)
        a = []
        for id_test in uids:
            a.append(eif.compute(int(id_test)))
        print("a:", a)
        ympred = a[self.ym]
        print("max y prediction", ympred)
        rank = sorted(a, reverse=True).index(ympred)
        print("rank:", rank)
        assert len(a) == uids.shape[0]

if __name__ == "__main__":
    unittest.main()


