import sys
sys.path.insert(0, "/export/a16/xzhan138/gbhpo")

import os
import json
import numpy as np
import time
import random
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.DEBUG)

from gbopt.fmin.gbopt import gb_opt
from gbopt.graph.graph import Graph

import argparse

parser = argparse.ArgumentParser(description='Graph Based Hyperparameter Optimization for Nasbench-101.')
parser.add_argument('--label_prop_alg', default='hmn', type=str, nargs='?',
                    help="label propagation algorithm.")
parser.add_argument('--acquisition_func', default='eif', type=str, nargs='?',
                    help="acquisition function.")
parser.add_argument('--maximize_func', default='random', type=str, nargs='?',
                    help="maximization function.")
parser.add_argument('--initial_design', default='random', type=str, nargs='?',
                    help="Initial design function.")
parser.add_argument('--num_iterations', default=1000, type=int, nargs='?',
                    help="Number of iterations.")
parser.add_argument('--n_samples', default=50000, type=int, nargs='?',
                    help="Number of samples for maximize_func.")
parser.add_argument('--n_init', default=10, type=int, nargs='?',
                    help="Number of initial points.")
parser.add_argument('--edit_distance', default=3, type=int, nargs='?',
                    help="Find neighbours of each node on the graph within edit distance x.")
parser.add_argument('--num_runs', default=500, type=int, nargs='?',
                    help="Number of repeat runs.")
parser.add_argument('--num_threads', default=60, type=int, nargs='?',
                    help="Number of threads for parallel running.")
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help="Directory for output files.")
parser.add_argument('--data_dir', type=str, nargs='?', help="Directory for data files.")
args = parser.parse_args()


print("Reading data files and building the graph...")
data_dir = args.data_dir
X_path = os.path.join(data_dir, "X")
Y_path = os.path.join(data_dir, "Y")
W_path = os.path.join(data_dir, "W")
I_path = os.path.join(data_dir, "I")

print("Reading X...")
with open(X_path) as f:
    X_ = f.readlines()
X = np.array([eval(i) for i in X_])

print("Reading Y...")
with open(Y_path) as f:
    trueY = f.readlines()
trueY = [eval(i) for i in trueY]

print("Reading W...")
with open(W_path) as f:
    W_ = f.readlines()
W = np.array([eval(i) for i in W_])

print("Reading I...")
with open(I_path) as f:
    I_ = f.readlines()
I = np.array([eval(i) for i in I_])

print("Assert len...")
assert len(W) == len(X), "The number of entries of affinity matrix should be equal to node count."
assert len(I) == len(X), "The number of entries of neighborhood matrix should be equal to node count."

Y = np.zeros(trueY.shape)
print("Building the graph...")
graph = Graph(X, Y, I, W)

print("Creating the dir...")
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

print("Optimizing...")
def objective_function(x):
    return trueY[X_.index(str(list(x))+"\n")]

def run_gb_opt(run_id):
    print("Run_id:", run_id)
    output_path = os.path.join(args.output_path, "run_" + str(run_id))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rd = random.randint(0, int(time.time()))
    rng = np.random.RandomState(rd)
    print("rd:", rd)

    results = gb_opt(graph, objective_function, label_prop_alg=args.label_prop_alg,
                     acquisition_func=args.acquisition_func, maximize_func=args.maximize_func,
                     n_samples=args.n_samples, num_iterations=args.num_iterations,
                     initial_design=args.initial_design, n_init=args.n_init, rng=rng,
                     output_path=output_path)

    with open(os.path.join(output_path, "results"), "w") as f:
        json.dump(results, f)

p = Pool(args.num_threads)
p.map(run_gb_opt, range(args.num_runs))
