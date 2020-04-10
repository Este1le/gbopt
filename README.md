# Graph-Based Optimization

This repository contains code of a graph-based optimization method. It is an extension of [graph-based semi-supervised regression](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf) for optimization problem.

To use this method, first define a grid of configurations (e.g. hyperparamter configurations for hyperparameter optimization) and build a graph upon them. Then specify the choice of label propagation rule and acquisition function. 

Refer to `experiment_scripts/run_gbopt.py` for an example of usage.

One example of the tabular datasets that can be used with the graph-based hyperparameter optimization is available [here](https://github.com/Este1le/hpo_nmt.git). It is a dataset for hyperparameter optimization of neural machine translation models. 

## Citation
```
@InProceedings{zhang-duh-nmthpo20,
	       author={Zhang, Xuan and Duh, Kevin},
	       title={Reproducible and Efficient Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems},
	       booktitle={Transactions of the Association for Computational Linguistics},
	       year={2020}
}
```
