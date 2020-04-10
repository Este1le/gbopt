'''
Code reference: https://github.com/automl/RoBO/blob/master/robo/solver/bayesian_optimization.py
'''
import os
import time
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

class GBOPT:

    def __init__(self, graph, objective_func, label_prop_alg,
                 acquisition_func, maximize_func,
                 initial_design, initial_points=3,
                 output_path=None, rng=None):
        """
        Implementation of the Graph-Based Optimization that uses an acquisition function
        and a label propagation algorithm to optimize a given objective function.
        This module keeps track of additional information such as runtime, optimization overhead,
        evaluated points and saves the output in a json file.

        Parameters
        ----------
        graph: GraphObject
            The graph defined by features and labels of nodes, and an adjacency matrix.
        objective_func:
            Function handle of for the objective function.
        acquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        maximize_func: BaseMaximizerObject
            The optimizers that maximize the acquisition function.
        label_prop_alg: LabelPropagationAlgObject
            The label propagation algorithm for graph-based semi-supervised learning.
        initial_design: function
            Function that returns some points which will be evaluated before the optimization loop is started.
            This allows to initialize the graph.
        initial_points: int
            Defines the number of initial points that are evaluated before the actual optimization.
        output_path: string
            Specifies the path where the intermediate output after each iteration will be saved.
            If None no output will be saved to disk.
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

        self.graph = graph
        self.objective_func = objective_func
        self.label_prop_alg = label_prop_alg
        self.acquisition_func = acquisition_func
        self.maximize_func = maximize_func
        self.initial_design = initial_design
        self.initial_points = initial_points
        self.output_path = output_path

        self.predictions = []

        self.incumbents = []
        self.incumbents_values = []

        self.time_func_evals = [] # time of each objective function evaluation
        self.time_overhead = [] # GBHPO time overhead
        self.runtime = [] # the overall runtime for each iteration (approximately equals to the sum of
                          # time_func_eval and time_overhead)
        self.start_time = time.time()

    def run(self, num_iterations=1000, lids_init=None, Y_init=None):
        """
        The main Graph-Based Hyperparameter Optimization loop.

        Parameters
        ----------
        num_iterations: int
            The number of iterations.
        lids_init: np.ndarray(l,)
            indices of initial points that are already evaluated.
        Y_init: np.ndarray(l,1)
            Function values of the already initial points.

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            function value of the incumbent
        """

        if lids_init is None:

            # Initial design
            start_time_overhead = time.time()
            lids_init = self.initial_design(self.graph.X,
                                       self.initial_points,
                                       rng=self.rng)
            time_overhead = (time.time() - start_time_overhead) / self.initial_points

            for i, lid in enumerate(lids_init):
                x = self.graph.X[lid]
                logger.info("Evaluate: %s", x)

                start_time = time.time()
                new_y = self.objective_func(x)
                self.time_func_evals.append(time.time() - start_time)

                self.graph.update(int(lid), new_y)
                self.time_overhead.append(time_overhead)

                logger.info("Configuration achieved a performance of %f in %f seconds",
                            new_y, self.time_func_evals[i])

                # Use best point seen so far as incumbent
                best_idx = np.argmax(self.graph.Y[self.graph.lids])
                incumbent = self.graph.X[self.graph.lids][best_idx]
                incumbent_value = self.graph.Y[self.graph.lids][best_idx]
                self.incumbents.append(incumbent.tolist())
                self.incumbents_values.append(incumbent_value)

                self.runtime.append(time.time() - self.start_time)

                if self.output_path is not None:
                    self.save_output(i)

        else:
            self.graph.update(lids_init, Y_init)

        # Main GBHPO loop
        for it in range(self.initial_points, num_iterations):
            #test:
            # print("it:", it)
            logger.info("Start iteration %d ...", it)

            if self.graph.lids.shape[0] == self.graph.X.shape[0]:
                break

            start_time = time.time()

            # Choose next point to evaluate
            new_id = self.choose_next(self.graph)
            new_x = self.graph.X[new_id]

            self.time_overhead.append(time.time() - start_time)
            logger.info("Optimization overhead was %f seconds", self.time_overhead[-1])
            logger.info("Next candidate %s", str(new_x))

            # Evaluate
            start_time = time.time()
            new_y = self.objective_func(new_x)
            self.time_func_evals.append(time.time() - start_time)

            logger.info("Configuration achieved a performance of %f", new_y)
            logger.info("Evaluation of this configuration took %f seconds", self.time_func_evals[-1])

            # Extend the data
            self.graph.update(new_id, new_y)

            # Estimate incumbent
            best_idx = np.argmax(self.graph.Y[self.graph.lids])
            incumbent = self.graph.X[self.graph.lids][best_idx]
            incumbent_value = self.graph.Y[self.graph.lids][best_idx]
            # test:
            # print("incumbent:", incumbent, "incumbent_value:", incumbent_value)

            self.incumbents.append(incumbent.tolist())
            self.incumbents_values.append(incumbent_value)
            logger.info("Current incumbent %s with estimated performance %f",
                        str(incumbent), incumbent_value)

            self.runtime.append(time.time() - self.start_time)

            if self.output_path is not None:
                self.save_output(it)

        logger.info("Return %s as incumbent with function value %f ",
                    self.incumbents[-1], self.incumbents_values[-1])

        return self.incumbents[-1], self.incumbents_values[-1]

    def choose_next(self, graph):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        graph: GraphObject
            The graph defined by nodes and an adjacency matrix.

        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """
        if self.graph.lids is None:
            lid = int(self.initial_design(graph.X, 1, rng=self.rng))
        else:
            logger.info("Label propagation through the graph...")
            t = time.time()
            self.label_prop_alg.label_propagate(graph)
            logger.info("Time to propagate the labels: %f", (time.time() - t))
            preds = graph.Y.tolist()
            self.predictions.append(preds)

            logger.info("Maximize acquisition function...")
            t = time.time()
            self.acquisition_func.update(graph)
            lid = self.maximize_func.maximize(graph)
            logger.info("Time to maximize the acquisition function: %f", (time.time() - t))

        return lid

    def save_output(self, it):

        data = dict()

        data["optimization_overhead"] = self.time_overhead[it]
        data["runtime"] = self.runtime[it]
        data["incumbent"] = self.incumbents[it]
        data["incumbent_value"] = self.incumbents_values[it].tolist()
        data["time_func_eval"] = self.time_func_evals[it]
        data["overhead"] = self.time_overhead[it]
        data["iteration"] = it
        if it >= self.initial_points:
            data["perdictions"] = self.predictions[it-self.initial_points]
        data["labeled_indices"] = self.graph.lids.tolist()

        json.dump(data, open(os.path.join(self.output_path, "iter_%d.json" % it), "w"))
