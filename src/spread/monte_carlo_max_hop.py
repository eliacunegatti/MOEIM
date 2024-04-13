import networkx as nx
import random
import numpy as np

""" Spread models """

""" Simulation of approximated spread for Independent Cascade (IC) and Weighted Cascade (WC). 
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""


def IC_model(G, a, p, max_hop, random_generator):  # a: the set of initial active nodes
	# p: the system-wide probability of influence on an edge, in [0,1]
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False

	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:  # G.neighbors follows A-B and A->B (successor) edges
				prob = random_generator.random()  # in the range [0.0, 1.0)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1

	return len(A)


def WC_model(G, a, max_hop, random_generator):  # a: the set of initial active nodes
	# each edge from node u to v is assigned probability 1/in-degree(v) of activating v
	A = set(a)  # A: the set of active nodes, initially a
	B = set(a)  # B: the set of nodes activated in the last completed iteration
	converged = False

	if nx.is_directed(G):
		my_degree_function = G.in_degree
	else:
		my_degree_function = G.degree

	while (not converged) and (max_hop > 0):
		nextB = set()
		for n in B:
			for m in set(G.neighbors(n)) - A:
				prob = random_generator.random()  # in the range [0.0, 1.0)
				p = 1.0 / my_degree_function(m)
				if prob <= p:
					nextB.add(m)
		B = set(nextB)
		if not B:
			converged = True
		A |= B
		max_hop -= 1

	return len(A)

def LT_model(G, a, max_hop, degrees, random_generator):
    A = set(a)
    B = set(a)
    converged = False

    threshold = {node: np.random.uniform(low=0.3, high=0.6) for node in G.nodes()}
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    activate = {node: len(neighbor_sets[node].intersection(A)) for node in G.nodes()}

    time = 0
    while not converged and (max_hop > 0):
        time += 1
        nextB = set()
        S = []

        for n in B:
            for m in neighbor_sets[n] - A - set(S):
                S.append(m)
                if activate[m] * degrees[m] > threshold[m]:
                    nextB.add(m)
        for m in nextB:        
            for t in neighbor_sets[m] - A:
                activate[t] += 1

        B = nextB
        if not B:
            converged = True
        A |= B
        max_hop -= 1
    
				    	
    return len(A)
def MonteCarlo_simulation_max_hop(G, A, p, no_simulations, model, max_hop, random_generator=None):
	"""
	calculates approximated influence spread of a given seed set A, with
	information propagation limited to a maximum number of hops
	example: with max_hops = 2 only neighbours and neighbours of neighbours can be activated
	:param G: networkx input graph
	:param A: seed set
	:param p: probability of influence spread (IC model)
	:param no_simulations: number of spread function simulations
	:param model: propagation model
	:param max_hops: maximum number of hops
	:return:
	"""
	if random_generator is None:
		random_generator = random.Random()

	results = []

	if model == 'WC':
		for i in range(no_simulations):
			results.append(WC_model(G, A, max_hop, random_generator))
	elif model == 'IC':
		for i in range(no_simulations):
			results.append(IC_model(G, A, p, max_hop, random_generator))
	elif model == 'LT':
		if nx.is_directed(G):
			degree_list = {node: 1 / G.in_degree(node) if G.in_degree(node) != 0 else 1 for node in G.nodes()}
		elif nx.is_directed(G) == False:
			degree_list = {node: 1 / G.degree(node) for node in G.nodes()}
		for i in range(no_simulations):
			results.append(LT_model(G, A, max_hop, degree_list,random_generator))

	return (np.mean(results), np.std(results))


