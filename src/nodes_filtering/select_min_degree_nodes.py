"""
Selection of nodes according to their degree
"""
import networkx as nx

def filter_best_nodes(G, min_degree, nodes=None):
	"""
	selects nodes with degree at least high as min_degree
	:param G:
	:param min_degree:
	:return:
	"""
	if nx.is_directed(G):
		my_degree_function = G.out_degree
	else:
		my_degree_function = G.degree
	if nodes is None:
		nodes = list(G.nodes())
	min_degree_nodes = nodes.copy()
	if min_degree > 0:
		for node in nodes:
			if my_degree_function(node) < min_degree:
				min_degree_nodes.remove(node)

	return min_degree_nodes

