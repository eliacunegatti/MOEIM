from networkx.generators import intersection
import numpy
import random
import networkx as nx
import numpy as np
from scipy import spatial
""" Spread models """

""" Simulation of spread for Independent Cascade (IC) and Weighted Cascade (WC). 
	Suits (un)directed graphs. 
	Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
"""

''''
Added time inside the cycle of the various models of propagation with the purpose to keep track of how much time it takes the propagation to converge to the optimal solution.
'''

## to re-code better
def LT_model(G, a, degrees, communities, bound_communities, max_time, lower_bound_comm, random_generator):
    A = set(a)
    B = set(a)
    converged = False

    threshold = {node: np.random.uniform(low=0.3, high=0.6) for node in G.nodes()}
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    activate = {node: len(neighbor_sets[node].intersection(A)) for node in G.nodes()}

    time = 0
    if max_time == None:
        max_time = np.inf

    while not converged and time < max_time:
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
        else:
            time += 1

        A |= B


    check = [len(set(item).intersection(A - set(a))) / len(item) for item in communities]
    #print('TIme', time)
    if sum(check) > 0:
        comm_ratio = np.array(check) / sum(np.array(check))
        return len(A), spatial.distance.jensenshannon(comm_ratio, bound_communities)/lower_bound_comm, time
    else:
        return len(A), 1, time

def IC_model(G, a, p, degrees,communities,bound_communities,max_time,  lower_bound_comm, random_generator):   
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    time = 0    
    if max_time == None:
        max_time = np.inf

    while not converged and time < max_time:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A: # G.neighbors follows A-B and A->B (successor) edges
                prob = random_generator.random() # in the range [0.0, 1.0)
                if prob <= p:   
                    nextB.add(m)        
        B = set(nextB)
        if not B:
            converged = True
        else:
            time += 1
        A |= B
    check = [len(set(item).intersection(A - set(a))) / len(item) for item in communities]
    if sum(check) > 0:
        comm_ratio = np.array(check) / sum(np.array(check))
        return len(A), spatial.distance.jensenshannon(comm_ratio, bound_communities)/lower_bound_comm, time
    else:
        return len(A), 1, time

def WC_model(G, a, degrees,communities,bound_communities,max_time,  lower_bound_comm,  random_generator):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    time = 0    
    if max_time == None:
        max_time = np.inf
    while not converged and time < max_time:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random() # in the range [0.0, 1.0)
                p = degrees[m]
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        else:
            time += 1
        A |= B  

    check = [len(set(item).intersection(A - set(a))) / len(item) for item in communities]
    if sum(check) > 0:
        comm_ratio = np.array(check) / sum(np.array(check))
        return len(A), spatial.distance.jensenshannon(comm_ratio, bound_communities)/lower_bound_comm, time
    else:
        return len(A), 1, time


def MonteCarlo_simulation(G, A, p,degrees, no_simulations, model, communities, bound_communities, max_time, low_bound, random_generator=None):
    if random_generator is None:
        random_generator = random.Random()
        random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable; TODO evaluate computational cost

    results = []
    comm_list = []
    times = []
    if model == 'WC':
        for i in range(no_simulations):
            res, comm, time = WC_model(G, A,degrees=degrees, communities=communities,bound_communities = bound_communities,max_time=max_time, lower_bound_comm= low_bound,  random_generator=random_generator)
            comm_list.append(comm)
            results.append(res)
            times.append(time)
    elif model == 'LT':
        for i in range(no_simulations):
            res, comm , time= LT_model(G, A, degrees=degrees, communities=communities,bound_communities = bound_communities,max_time=max_time,  lower_bound_comm= low_bound, random_generator=random_generator)
            comm_list.append(comm)
            results.append(res)
            times.append(time)
    elif model == 'IC':
        for i in range(no_simulations):
            res, comm , time= IC_model(G, A,p=p ,degrees=degrees, communities=communities,bound_communities = bound_communities,max_time=max_time,  lower_bound_comm= low_bound, random_generator=random_generator)
            comm_list.append(comm)
            results.append(res)
            times.append(time)
    return (numpy.mean(results), numpy.std(results), float(numpy.mean(comm_list)), int(numpy.mean(times)))


