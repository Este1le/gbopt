import numpy as np
from gbopt.maximizers.base_maximizer import BaseMaximizer

class RandomSampling(BaseMaximizer):

    def __init__(self, acquisition_func, n_samples=-1, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        n_samples: int
            The number of points to evaluate.
        rng: np.random.RandomState
            Random number generator.
        """
        self.n_samples = n_samples
        super(RandomSampling, self).__init__(acquisition_func, rng)

    def maximize(self, graph):
        """
        Maximizes the given acquisition function.

        Parameters
        ----------
        graph: GraphObject
            The graph defined by nodes and an adjacency matrix.

        Returns
        -------
        int
            Index of point with highest acquisition value.
        """
        uids = np.delete(np.arange(graph.Y.shape[0]), graph.lids)
        if self.n_samples == -1:
            rand = uids
        else:
            rand = self.rng.choice(uids, self.n_samples, replace=False)

        af = np.array([self.acquisition_func(int(i)) for i in rand])


        return int(uids[np.argmax(af)])
