import numpy as np

from gbopt.acquisition_functions.base_acquisition import BaseAcquisitionFunction

class Random(BaseAcquisitionFunction):

    def __init__(self):
        """
        Randomly assign acquisition values to nodes of the graph.

        Parameters
        ----------
        graph:GraphObject
            The graph defined by nodes and adjacency matrix.
        """
        super(Random, self).__init__()

    def compute(self, i):
        return np.random.random(1)[0]

