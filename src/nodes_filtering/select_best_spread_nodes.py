import numpy as np
import random
from scipy import stats
import sys

import src.utils


def sample_mc(node, num_samples, sampling_func):
	"""
	samples num_samples using sampling func
	:param node:
	:param num_samples:
	:param sampling_func:
	:return:
	"""
	values = []
	for _ in range(num_samples):
		s = sampling_func(A=[node])
		if isinstance(s, list) or isinstance(s, tuple):
			s = s[0]
		values.append(s)
	return values


def compute_error(sample, alpha):
	"""
	computes the confidence interval allowed error
	:param sample:
	:param alpha:
	:return:
	"""
	n = len(sample)
	std = np.std(sample)
	t = stats.t.ppf(1 - alpha / 2, n - 1)
	error = (t * std) / np.sqrt(n)
	return error


def _best_nodes(nodes_samples, k, nodes_errors):
	"""
	find best k nodes + all the other nodes which have comparable values
	:param nodes_samples:
	:return:
	"""
	spreads_arr = [nodes_samples[node] for node in nodes_samples]
	errors = [nodes_errors[node] for node in nodes_samples]
	spreads = [np.mean(sample) for sample in spreads_arr]
	spreads = np.array(spreads)
	errors = np.array(errors)
	nodes = [node for node in nodes_samples]
	nodes = np.array(nodes)
	sorted_idx = np.argsort(spreads)
	spreads = spreads[sorted_idx]
	errors = errors[sorted_idx]
	nodes = nodes[sorted_idx]
	k_th_node = nodes[-k]
	threshold = spreads[-k] - nodes_errors[k_th_node]
	best = nodes[-k:]

	upper_bounds = spreads + errors
	not_comparable = nodes[upper_bounds>threshold]

	best_1 = np.union1d(best, not_comparable)

	return best_1


def _evaluate_spreads_with_allowed_error(nodes, allowed_error, alpha, sampling_func, nodes_samples=None):
	"""
	samples a necessary quantity of data points in order to guarantee a maximum allowed error on the mean
	:param nodes: list of nodes to evaluate
	:param allowed_error: percentage of allowed error on the mean
	:param alpha: 1-confidence for calculating confidence interval with student t value
	:param sampling_func: function that computes monte carlo spread sampling
	:param nodes_samples: dictionary with precomputed nodes samples to update
	:return: returns a dictionary with spread samples for each node {node_id: list_of_spread_values}, and
			 a dictionary with confidence intervals on these values {node_id: error}, which means
			 the spread value of the node is contained in the interval [mean-error, mean+error] with confidence
			 1-alpha
	"""
	if nodes_samples is None or nodes_samples == {}:
		nodes_samples = {}
		n = 5
	else:
		n = 3

	nodes_errors = {}

	for j, node in enumerate(nodes):
		sys.stdout.write("Node {}/{} \r".format(j, len(nodes)))

		if node not in nodes_samples.keys():
			nodes_samples[node] = []
		values = sample_mc(node, n, sampling_func)
		for value in values:
			nodes_samples[node].append(value)
		ok = False
		iters = 0
		while not ok:
			mean = np.mean(nodes_samples[node])
			margin1 = mean * allowed_error

			# compute the statistical confidence
			diff = compute_error(nodes_samples[node], alpha)
			nodes_errors[node] = abs(diff)
			if margin1 < abs(diff) and len(nodes_samples[node]) < 100:
				n += max(int(n / (iters + 2)), 1)
				for _ in range(n-len(nodes_samples[node])):
					nodes_samples[node].append(sample_mc(node, 1, sampling_func)[0])
				iters += 1
			else:
				ok = True
	return nodes_samples, nodes_errors


def filter_best_nodes(G, n, allowed_n_error, sampling_func):
	"""
	selects best n nodes according to the monte carlo sampling spread function
	:param G: input nx graph
	:param n: number of best nodes
	:param allowed_n_error: allowed error percentage on n value
	:param sampling_func: monte carlo spread sampling function
	:return:
	"""
	allowed_err = 0.8
	best = np.array(G.nodes)
	nodes_samples = {}
	iter = 1
	while len(best) > n * (1 + allowed_n_error) and allowed_err > 0.1:
		print("Nodes filtering iter: {}".format(iter))
		nodes_samples, nodes_errors = _evaluate_spreads_with_allowed_error(best, allowed_err, 0.05,
																		  sampling_func, nodes_samples)
		best_samples = {}
		for node in best:
			best_samples[node] = nodes_samples[node]
		best = _best_nodes(best_samples, n, nodes_errors)
		allowed_err -= 0.1
		allowed_err = round(allowed_err, 4)
		iter += 1

	return list(best)

