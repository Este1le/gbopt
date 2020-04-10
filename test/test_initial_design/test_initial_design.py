import unittest
import numpy as np
from gbopt import initial_design
from test.dummy_graph import graph1d
from test.dummy_graph import graph2d

class TestInitialDesign(unittest.TestCase):
    def setUp(self):
        self.n_points = 5
        self.X1d = graph1d.X
        self.X2d = graph2d.X

    def test_init_random_choice_1d(self):

        ids = initial_design.init_random_choice(self.X1d, self.n_points)

        assert ids.shape == (self.n_points, )
        assert np.all(np.min(ids, axis=0) >= 0)
        assert np.all(np.max(ids, axis=0) <= 100)

    def test_init_random_choice_2d(self):

        ids = initial_design.init_random_choice(self.X2d, self.n_points)

        assert ids.shape == (self.n_points, )
        assert np.all(np.min(ids, axis=0) >= 0)
        assert np.all(np.max(ids, axis=0) <= 120)

if __name__ == "__main__":
    unittest.main()
