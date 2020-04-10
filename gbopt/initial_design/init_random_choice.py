'''
Code reference: https://github.com/automl/RoBO/blob/master/robo/initial_design/init_random_uniform.py
'''
import numpy as np

def init_random_choice(X, n_points, rng=None):
    """
    Samples n_points data points from X randomly.

    Parameters
    ----------
    X: np.ndarray (N,D)
        Sample pool.
    n_points: int
        The number of initial data points.
    rng: np.random.RandomState
        Random number generator.

    Returns
    -------
    np.ndarray(l, )
        The indices of initial design data points.
    """

    if rng is None:
        rng = np.random.RandomState(np.random.randint(100000))

    return rng.choice(X.shape[0], n_points, replace=False)
