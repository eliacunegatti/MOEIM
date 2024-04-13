import networkx as nx
import random
import numpy
import math
import time
import numpy as np
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", A is followed by B", "A is trusted by B". 

def IC_model(G, a, p, max_time,random_generator):              # a: the set of initial active nodes
                                    # p: the system-wide probability of influence on an edge, in [0,1]
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    t = 0
    while not converged and t < max_time:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()	# in the range [0.0, 1.0)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        else:
            t += 1
        A |= B

    return len(A), t

def WC_model(G, a, max_time,random_generator):                 # a: the set of initial active nodes
                                    # each edge from node u to v is assigned probability 1/in-degree(v) of activating v
    A = set(a)                      # A: the set of active nodes, initially a
    B = set(a)                      # B: the set of nodes activated in the last completed iteration
    converged = False
    t = 0
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree
    while not converged and t < max_time:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()	# in the range [0.0, 1.0)
                p = 1.0/my_degree_function(m)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        else:
            t += 1
        A |= B

    return len(A), t


def LT_model(G, a, degrees, max_time,random_generator):
    A = set(a)
    B = set(a)
    converged = False

    threshold = {node: np.random.uniform(low=0.3, high=0.6) for node in G.nodes()}
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    activate = {node: len(neighbor_sets[node].intersection(A)) for node in G.nodes()}
    t = 0
    while not converged and t < max_time:
        nextB = set()
        S = []

        for n in B:
            for m in neighbor_sets[n] - A - set(S):
                S.append(m)
                if activate[m] * degrees[m] > threshold[m]:
                    nextB.add(m)
        for m in nextB:        
            for t_ in neighbor_sets[m] - A:
                activate[t_] += 1
        B = nextB
        if not B:
            converged = True
        else:
            t += 1
        A |= B

    return len(A), t

# evaluates a given seed set A
# simulated "no_simulations" times
# returns a tuple: the mean, stdev, and 95% confidence interval
def evaluate(G, A, p, no_simulations, model, max_time, random_generator=None):

    results = []
    time = []
    if model == 'WC':
        for i in range(no_simulations):
            res = WC_model(G, A, max_time, random_generator) 
            results.append(res[0])
            time.append(res[1])
    elif model == 'IC':
        for i in range(no_simulations):
            res = IC_model(G, A,p, max_time,random_generator) 
            results.append(res[0])
            time.append(res[1])
    elif model == 'LT':
        if nx.is_directed(G):
            degree_list = {node: 1 / G.in_degree(node) if G.in_degree(node) != 0 else 1 for node in G.nodes()}
        elif nx.is_directed(G) == False:
            degree_list = {node: 1 / G.degree(node) for node in G.nodes()}
        for i in range(no_simulations):
            res = LT_model(G, A,degree_list,max_time,random_generator) 
            results.append(res[0])
            time.append(res[1])      

    return numpy.mean(results), numpy.std(results), numpy.sum(time)



