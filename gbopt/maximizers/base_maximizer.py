"""
Code reference: https://github.com/automl/RoBO/blob/master/robo/maximizers/base_maximizer.py
"""
import numpy as np

class BaseMaximizer(object):

    def __init__(self, acquisition_func, rng=None):
        """
        Interface for optimizers that maximizing the acquisition function.

        Parameters
        ----------
        aquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        rng: np.random.RandomState
            Random number generator
        """
        self.acquisition_func = acquisition_func
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

    def maximize(self):
        pass
