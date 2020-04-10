import numpy as np

class Graph:
    def __init__(self, X, Y, I, W, lids=None):
        """
        Parameters
        ----------
        X: np.ndarray(N,D)
            The nodes of the graph.
        Y: np.ndarray(N,)
            The labels.
        I: np.ndarray(N, i)
            Neighbour indices of each node.
        W: np.ndarray(N, i)
            Affinity matrix.
        lids: np.ndarray(l,)
            Indices of labeled nodes.
        """
        self.X = X
        self.Y = Y
        self.lids = lids
        self.I = I
        self.W = W

    def update(self, ids, new_Y):
        if type(ids) == int:
            ids = np.array([ids])
        else:
            ids = np.array(ids)
        if self.lids is None:
            self.lids = ids
        else:
            self.lids = np.concatenate((self.lids, ids))
        self.Y[ids] = new_Y

    def remove(self, i):
        self.lids = np.delete(self.lids, np.argwhere(self.lids==i))
