import numpy as np

class HMN:
    def __init__(self, graph):
        self.graph = graph
        self.W = np.zeros((self.graph.X.shape[0], self.graph.X.shape[0]))
        for i in range(self.graph.X.shape[0]):
            self.W[i][self.graph.I[i]] = self.graph.W[i]
        self.D = np.diag(np.sum(self.W, axis=1))
        self.delta = self.D - self.W

    def label_propagate(self, graph):
        uids = np.delete(np.arange(graph.Y.shape[0]), graph.lids)
        delta_uu = self.delta[np.ix_(uids, uids)]
        delta_ul = self.delta[np.ix_(uids, graph.lids)]
        delta_uu_inv = np.linalg.inv(delta_uu)
        graph.Y[uids] = -delta_uu_inv.dot(delta_ul).dot(graph.Y[graph.lids])
