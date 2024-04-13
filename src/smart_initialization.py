import networkx as nx
import community
import numpy as np
from sklearn.cluster import SpectralClustering
import random

def max_centrality_individual(k, G, centrality_metric="degree"):
	"""
	returns k nodes with the highest centrality metric
	:param k: seed set size
	:param G: networkx graph
	:param centrality_metric: centrality metric string, to be chosen from "degree", "eigenvector", "katz", "closeness",
							  "betweenness", "second_order"
	:return:
	"""
	if centrality_metric == "degree":
		nodes_centrality = nx.degree_centrality(G)
	elif centrality_metric == "eigenvector":
		nodes_centrality = nx.eigenvector_centrality_numpy(G)
	elif centrality_metric == "katz":
		nodes_centrality = nx.katz_centrality_numpy(G)
	elif centrality_metric == "closeness":
		nodes_centrality = nx.closeness_centrality(G)
	elif centrality_metric == "betweenness":
		nodes_centrality = nx.betweenness_centrality(G)
	elif centrality_metric == "second_order":
		nodes_centrality = nx.second_order_centrality(G)

	sorted_nodes_centrality = dict(sorted(nodes_centrality.items(), key=lambda nodes_centrality: nodes_centrality[1],
										  reverse=True))
	return list(sorted_nodes_centrality)[:k]


def degree_random(k, Graph, n, prng, nodes=None):
	"""
	Returns n individuals containing k nodes, chosen from graph with
	probabilities proportional to their degrees.
	"""
	individuals = []
	if nodes is None:
		nodes = Graph.nodes()
	nodes_degree = []
	all_nodes_degree = nx.degree_centrality(Graph)
	for node in nodes:
		nodes_degree.append(all_nodes_degree[node])
	nodes_degree = np.array(nodes_degree)
	nodes = np.array(nodes)
	for _ in range(n):
		probs = nodes_degree.copy()
		nodes_ = nodes.copy()
		new_indiv = []
		range_k = random.randint(1, k) #k if fix seet is needed
		for _ in range(range_k):
			if len(probs) > 0:
				new_node = prng.choices(nodes_, probs)[0]
				probs = probs[nodes_!=new_node]
				nodes_ = nodes_[nodes_!=new_node]
				new_indiv.append(new_node)
		individuals.append(new_indiv)
	return individuals


class Community_initialization:
	"""
	class containing methods for smart initialization using community detection algorithms
	"""
	def __init__(self, G, random_seed=None, method="louvain", n_clusters=None):
		"""
		class initialization: run community detection and store the results for future reuse
		:param G: networkx graph
		:param random_seed:
		:param method: method to use for community detection, choose from ["louvain", "spectral_clustering"]
		:param n_clusters: number of clusters ( communities ), to be specified only when spectral_clustering community
			detection method is used
		"""

		self.G_original = G.copy().to_undirected()
		self.G = self.G_original.copy()

		self.prng = np.random.RandomState(random_seed)

		if method == "louvain":
			self.comm_part = community.best_partition(self.G, random_state=self.prng)
			comm_idxs = set(self.comm_part.values())
			# dictionary containing community indexes as keys and graph labels as values
			self.communities_original = dict()
			for comm_idx in comm_idxs:
				self.communities_original[comm_idx] = np.array([k for k, v in self.comm_part.items() if v == comm_idx])

		elif method == "spectral_clustering":
			if n_clusters is None:
				raise Exception('Number of clusters not specified')
			sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100, assign_labels='discretize', 
									random_state=random_seed)
			adj = nx.to_scipy_sparse_matrix(G)
			sc.fit(adj)
			comm_idxs = set(sc.labels_)
			# dictionary containing community indexes as keys and graph labels as values
			self.communities_original = dict()
			for comm_idx in comm_idxs:
				nodes = np.array(G.nodes())
				self.communities_original[comm_idx] = np.array(nodes[sc.labels_==comm_idx])

			# substitute labels with label enumerations
			tmp = dict()
			for i, k in enumerate(self.communities_original):
				tmp[i] = self.communities_original[k]
			self.communities_original = tmp
		self.degree = None

	def _get_random_degree_node(self, nodes, n):
		"""
		returns n randomly selected nodes from the given nodes, a probability of a node to be chosen is proportional
		to its degree
		:param nodes:
		:return:
		"""
		# if self.degree is None:
		self.degree = nx.degree_centrality(self.G)

		nodes_degrees = np.array([self.degree[node] for node in nodes])

		probabilities = nodes_degrees / sum(nodes_degrees)
		best_degree_nodes = self.prng.choice(nodes, n, p=probabilities)
		return best_degree_nodes

	def get_comm_members_random(self, n, k, degree=False):
		"""
		returns n members randomly chosen from communities, the probability to be taken
		from a community is proportional to its size
		:param n:
		:param degree: if true, for each randomly selected community the probability to select a node is
		proportional to its degree
		:return:
		"""
		result = []
		for i in range(n):
			indiv = []
			# reset values
			self.G = self.G_original.copy()
			self.communities = self.communities_original.copy()
			for _ in range(k):
				# recalculate communities and probabilities
				max = len(self.communities)
				probabilites = np.zeros(len(self.communities))
				tot_len = 0
				for i in self.communities:
					len_comm = len(self.communities[i])
					tot_len += len_comm
					probabilites[i] = len_comm
				probabilites /= tot_len

				# choose a community
				idx = self.prng.choice(max, 1, p=probabilites)[0]
				# choose a node inside the community
				if degree:
					comm_random_indiv = self._get_random_degree_node(self.communities[idx], 1)[0]
				else:
					comm_random_indiv = self.prng.choice(self.communities[idx])
				indiv.append(comm_random_indiv)

				# remove the node from communities and graph
				self.G.remove_node(comm_random_indiv)
				self.communities[idx] = np.delete(self.communities[idx], np.where(self.communities[idx] == comm_random_indiv))
			result.append(indiv)
		return result

