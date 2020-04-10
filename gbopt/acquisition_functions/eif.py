import numpy as np

from gbopt.acquisition_functions.base_acquisition import BaseAcquisitionFunction

class EIF(BaseAcquisitionFunction):

    def __init__(self, graph, label_prop_alg):
        r"""
        Computes for a given x the expected influence as acquisition function value.

        :math:`EIF(x) := (1 - f(x)) \sum_{i=1}^n (1 - f^(+(x,0)))(i)
                          + f(x) \sum_{i=1}^n f^(+(x),1)(i)`, where
        :math:`f(i)` are all scaled to [0,1].

        Parameters
        ----------
        graph: GraphObject
            The graph defined by nodes and adjacency matrix.
        label_prop_alg: BaseAcquisitionObject
            The label propagation algorithm for graph-based semi-supervised learning.
        """
        super(EIF, self).__init__(graph, label_prop_alg)

    def _normalize(self):
        """
        Scale Y to the range [0,1].
        Set the max labeled point as a pivot, and set its value to 1.
        Scale the predictions of unlabeled points that is greater than pivot to [0.5, 1].
        Scale the predictions of unlabeled points that it smaller than pivot to [0. 0.5).
        """
        lids = self.graph.lids
        Y = self.graph.Y
        n = Y.shape[0]
        uids = np.delete(np.arange(Y.shape[0]), lids)
        YU = Y[uids]
        YL = Y[lids]

        Y_norm = np.zeros((n))

        pivot = max(YL)
        pivot_id = lids[np.argmax(YL)]
        Y_norm[pivot_id] = 1

        YU_max = max(YU)
        YU_min = min(YU)

        for u in uids:
            if Y[u] >= pivot:
                Y_norm[u] = (Y[u] - pivot) / (YU_max - pivot) * 0.5 + 0.5
            else:
                Y_norm[u] = (pivot - Y[u]) / (pivot - YU_min) * 0.5

        return Y_norm

    def compute(self, i):
        """
        Computes the EIF value for a given point x.

        Parameters
        ----------
        i: int
            The index of input point where the acquisition function should be evaluated.

        Returns
        -------
        np.ndarray(1,1)
            Expected influence of x.
        """
        Y_norm = self._normalize()
        Y = self.graph.Y

        self.graph.Y = Y_norm
        y = self.graph.Y[i]
        # If the label of x is 0
        self.graph.update(i, 0)
        self.label_prop_alg.label_propagate(self.graph)
        uids = np.delete(np.arange(self.graph.Y.shape[0]), self.graph.lids)
        YU0 = self.graph.Y[uids]
        self.graph.remove(i)

        self.graph.Y = Y_norm
        # If the label of x is 1
        self.graph.update(i, 1)
        self.label_prop_alg.label_propagate(self.graph)
        YU1 = self.graph.Y[uids]
        self.graph.remove(i)

        self.graph.Y = Y

        return (1 - y) * np.sum(1 - YU0) + y * np.sum(YU1)
        #return y * np.sum(YU1)
