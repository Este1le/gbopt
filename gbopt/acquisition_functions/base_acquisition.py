'''
Code reference: https://github.com/automl/RoBO/blob/master/robo/acquisition_functions/base_acquisition.py
'''
import abc

class BaseAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, graph=None, label_prop_alg=None):
        """
        A base class for acquisition functions.

        Parameters
        ----------
        graph: GraphObject
            The graph defined by nodes and adjacency matrix.
        label_prop_alg: BaseAcquisitionObject
            The label propagation algorithm for graph-based semi-supervised learning.
        """
        self.graph = graph
        self.label_prop_alg = label_prop_alg

    def update(self, graph):
        """
        This method will be called if the graph is updated.

        Parameters
        ----------
        graph: GraphObject
            The graph defined by nodes and adjacency matrix.
        """

        self.graph = graph

    @abc.abstractmethod
    def compute(self, i):
        """
        Computes the acquisition function value for a given point x.
        This function has to be overwritten in a derived class.

        Parameters
        ----------
        i: int
            The index of input point where the acquisition function should be evaluated.
        """
        pass

    def __call__(self, x):
        return self.compute(x)
