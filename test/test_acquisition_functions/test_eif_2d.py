import unittest
import numpy as np
from test.dummy_graph import graph2d
from gbopt.acquisition_functions.eif import EIF
from gbopt.label_propagation_algorithms.hmn import HMN

class TestEIF2D(unittest.TestCase):

    def setUp(self):
        self.graph2d = graph2d
        print(self.graph2d.Y)
        lids2d = np.random.choice(graph2d.X.shape[0], 10, replace=False)
        uids = np.delete(np.arange(self.graph2d.Y.shape[0]), lids2d)
        self.ym = np.argmax(self.graph2d.Y[uids])
        Yl2d = graph2d.Y[lids2d]
        graph2d.Y = np.zeros((graph2d.Y.shape[0]))
        graph2d.update(lids2d, Yl2d)
        self.lp = HMN(self.graph2d)

    def test_compute(self):
        self.lp.label_propagate(self.graph2d)
        print(self.graph2d.Y)
        eif = EIF(self.graph2d, self.lp)
        uids = np.delete(np.arange(self.graph2d.Y.shape[0]), self.graph2d.lids)
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


