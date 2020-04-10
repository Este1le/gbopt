import logging
import numpy as np

from gbopt.solver.gbopt import GBOPT
from gbopt.maximizers.random_sampling import RandomSampling
from gbopt.acquisition_functions.eif import EIF
from gbopt.acquisition_functions.random import Random
from gbopt.label_propagation_algorithms.hmn import HMN
from gbopt.initial_design.init_random_choice import init_random_choice

logger = logging.getLogger(__name__)

def gb_opt(graph, objective_function, label_prop_alg="hmn",
           acquisition_func="eif", maximize_func="random", n_samples=-1,
           num_iterations=1000, initial_design="random",
           n_init=10, lids_init=None, Y_init=None, rng=None, output_path=None):
    """
    General interface for graph-based semi-supervised learning for global
    black box optimization problems.

    Parameters
    ----------
    graph: GraphObject
        The graph defined by features and labels of nodes, and an adjacency matrix.
    objective_func:
        Function handle of for the objective function.
    label_prop_alg: LabelPropagationAlgObject
        The label propagation algorithm for graph-based semi-supervised learning.
    acquisition_func: BaseAcquisitionFunctionObject
        The acquisition function which will be maximized.
    maximize_func: BaseMaximizerObject
        The optimizers that maximize the acquisition function.
    n_samples: int
        The number of points to evaluate for maximize_func.
        -1 is evaluating all the points on the graph.
    initial_design: function
        Function that returns some points which will be evaluated before the optimization loop is started.
        This allows to initialize the graph.
    initial_points: int
        Defines the number of initial points that are evaluated before the actual optimization.
    lids_init: np.ndarray(l,)
        Indices of initial points to warm start label propagation.
    Y_init: np.ndarray(l,1)
        Function values of the already initial points.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: np.random.RandomState
        Random number generator

    Returns
    -------
        dict with all results
    """

    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations."
    assert lids_init is None or lids_init in np.arange(graph.X.shape[0]), "The initial points has to be in the sample pool."

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    if label_prop_alg == "hmn":
        lp = HMN(graph)
    else:
        raise ValueError("'{}' is not a valid label propagation algorithm".format(label_prop_alg))

    if acquisition_func == "eif":
        af = EIF(graph, lp)
    elif acquisition_func == "random":
        af = Random()
    else:
        raise ValueError("'{}' is not a valid acquisition function".format(acquisition_func))

    if maximize_func == "random":
        max_func = RandomSampling(af, n_samples=n_samples, rng=rng)
    else:
        raise ValueError("'{}' is not a valid function to maximize the "
                         "acquisition function".format(maximize_func))

    if initial_design == "random":
        init_design = init_random_choice
    else:
        raise ValueError("'{}' is not a valid function to design the "
                         "initial points".format(initial_design))

    gbo = GBOPT(graph, objective_function, lp,
                af, max_func,
                initial_design=init_design, initial_points=n_init,
                rng=rng, output_path=output_path)

    x_best, f_max = gbo.run(num_iterations, lids_init, Y_init)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_max.tolist()
    # Best x along training
    results["incumbents"] = [inc for inc in gbo.incumbents]
    results["incumbents_values"] = [val for val in gbo.incumbents_values]
    results["runtime"] = gbo.runtime
    results["overhead"] = gbo.time_overhead
    results["labeled_indices"] = gbo.graph.lids.tolist()
    results["X"] = [x.tolist() for x in gbo.graph.X[gbo.graph.lids]]
    results["y"] = [y for y in gbo.graph.Y[gbo.graph.lids]]
    results["predictions"] = gbo.predictions

    return results
