import inspyred
import numpy as np


def get_nodes_without_repetitions(candidate, args):
	"""
	removes candidate nodes from the pool of nodes
	:param candidate:
	:param args:
	:return:
	"""
	nodes = args["_ec"].bounder.values.copy()
	for c in candidate:
		if c in nodes: nodes.remove(c)
	return nodes
    


def get_nodes_neighbours_without_repetitions(node, candidate, args):
	"""
	returns nodes neighbours without nodes in candidate
	:param node:
	:param candidate:
	:param args:
	:return:
	"""
	nodes = list(args["G"].neighbors(node))
	# avoid nodes repetitions
	for c in candidate:
		if c in nodes: nodes.remove(c)
	return nodes


def eval_fitness(seed_set, random, args):
	"""
	evaluates fitness of the seed set
	:param seed_set:
	:param random:
	:return:
	"""
	spread = args["fitness_function"](A=seed_set, random_generator=random)
	# if we are using monteCarlo simulations which returns mean and std
	if len(spread) > 0:
		spread = spread[0]
	return spread


@inspyred.ec.variators.mutator
def ea_global_random_mutation(prng, candidate, args):
	"""
	Randomly mutates one gene of the individual with one random node of the graph.
	"""
	#assure that candidate is a list

	if len(candidate) > 1:
		nodes = get_nodes_without_repetitions(candidate, args)

		mutatedIndividual = candidate.copy()
		# choose random gene
		gene = prng.randint(0, len(mutatedIndividual) - 1)
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]

		mutatedIndividual[gene] = mutated_node
		return mutatedIndividual
	else:
		return candidate

@inspyred.ec.variators.mutator
def ea_local_neighbors_random_mutation(prng, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors
	"""
	mutatedIndividual = candidate.copy()
	# choose random gene
	gene = prng.randint(0, len(mutatedIndividual) - 1)

	# choose among neighbours of the selected node
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		mutated_node = nodes[prng.randint(0, len(nodes) - 1)]
		mutatedIndividual[gene] = mutated_node
	else:
		# if we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(prng, [candidate], args)[0]

	return mutatedIndividual

@inspyred.ec.variators.mutator
def ea_local_neighbors_second_degree_mutation(random, candidate, args):
	"""
	randomly mutates one gene of the individual with one of it's neighbors, but according to second degree probability
	"""
	mutatedIndividual = candidate

	# choose random gene
	gene = random.randint(0, len(mutatedIndividual) - 1)
	# choose among neighbours of the selected node
	nodes = get_nodes_neighbours_without_repetitions(mutatedIndividual[gene], candidate, args)

	if len(nodes) > 0:
		# calculate second degree of each of the neighbors
		second_degrees = []
		for node in nodes:
			sec_degree = 0
			sec_degree += len(nodes)
			node_neighs = list(args["G"].neighbors(node))
			for node_neigh in node_neighs:
				# !very roughly approximated to reduce computation time, may include repetitions
				neighbors_of_neighbors = list(args["G"].neighbors(node_neigh))
				sec_degree += len(neighbors_of_neighbors)
			second_degrees.append(sec_degree)
		probs = np.array(second_degrees) / max(second_degrees)
		idx = random.choices(range(0, len(nodes)), probs)[0]
		mutatedIndividual[gene] = nodes[idx]
	else:
		# If we don't have neighbors to choose from, global mutation
		mutatedIndividual = ea_global_random_mutation(random, [candidate], args)[0]
	return mutatedIndividual


@inspyred.ec.variators.mutator
def ea_global_low_deg_mutation(random, candidate, args):
	"""
	the probability to select the gene to mutate depends on its degree
	"""

	nodes = get_nodes_without_repetitions(candidate, args)
	mutatedIndividual = candidate

	# choose random gene
	probs = []
	for node in mutatedIndividual:
		probs.append(args["G"].degree(node))

	probs = np.array(probs) / max(probs)
	probs = 1 - probs

	gene = random.choices(range(0, len(mutatedIndividual)), probs)[0]
	mutated_node = nodes[random.randint(0, len(nodes) - 1)]

	mutatedIndividual[gene] = mutated_node

	return mutatedIndividual








