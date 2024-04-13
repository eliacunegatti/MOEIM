
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import networkx as nx
import heapq as hq
import SNSigmaSim_networkx as SNSim
import time
from src.load import read_graph
import pandas as pd
from src.spread.monte_carlo import MonteCarlo_simulation as MonteCarlo_simulation
import numpy as np
from scipy import spatial
import copy
from pymoo.indicators.hv import Hypervolume
import argparse
import json

#code based on https://github.com/albertotonda/Influence-Maximization/blob/master/heuristics.py

##################### utils functions ######################

def get_fairness(args, A_set):
    check = []
    for item in args["communities"]:
        intersection = set.intersection(set(item),set(A_set))
        check.append(len(set(intersection))/len(set(item)))

    #print('Check', check)
    fairness = 0
    if sum(check) > 0:
        comm_ratio = np.array(check) / sum(np.array(check))
        fairness = spatial.distance.jensenshannon(comm_ratio, args["bound_communities"])/args["lower_bound_communities"]
    else:
        fairness = 1
    return fairness

def max_budget(G,k):

    if nx.is_directed(G):
        b = sorted([G.out_degree(i) for i in G.nodes()], reverse=True)
    elif nx.is_directed(G) == False:
        b = sorted([G.degree(i) for i in G.nodes()], reverse=True)
    b = sum(b[:k])

    return b

def minimize_obj(A):
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] = -float(A[i][j])
    return np.array(A)


def hypervolume_seed_spread(A, args):
    F = minimize_obj(copy.deepcopy(A))
	#A = np.array(A)
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    metric = Hypervolume(ref_point= np.array([0,0]),
						norm_ref_point=False,
						zero_to_one=False)
    hv = metric.do(F)
	#compute the hypervolume for list A knowing that the reference point is [1,1
    return hv

#hypervolume for seed and time
def hypervolume_with_one(A, args):
    F = minimize_obj(copy.deepcopy(A))
    #print(F[0].shape, F.shape)
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    metric = Hypervolume(ref_point= np.array([0,0,0]),
						norm_ref_point=False,
						zero_to_one=False)
    hv = metric.do(F)
    return hv




def get_graph(args):

    filename = 'graphs_cleaned/{0}.txt'.format(args["graph"])
    graph_name = str(os.path.basename(filename))
    graph_name = graph_name.replace(".txt", "")
    df = pd.read_csv(f'graph_communities/{args["graph"]}.csv', sep=',')
    comm  = [[] for i in range(len(set(df["comm"].to_list())))]
    nodes_ = df["node"].to_list()
    comm_ = df["comm"].to_list()
    print(len(comm))
    for i in range(len(nodes_)):
        comm[comm_[i]-1].append(nodes_[i])
    G = read_graph(filename,comm)

    return G, graph_name, comm

#######

############################ IM heuristics metrics: code based on https://github.com/albertotonda/Influence-Maximization/blob/master/heuristics.py ############################
def generalized_degree_discount(k, G, p_values, model, random_generator):
    if nx.is_directed(G):
        my_predecessor_function = G.predecessors
        my_degree_function = G.out_degree
    else:
        my_predecessor_function = G.neighbors
        my_degree_function = G.degree

    S = []
    GDD = {}
    t = {}

    for n in G.nodes():
        GDD[n] = my_degree_function(n)
        t[n] = 0

    for i in range(k):
        print('S', S, 'K', k)
        # select the node with current max GDD from V-S
        u = max(set(list(GDD.keys())) - set(S), key=(lambda key: GDD[key]))
        S.append(u)
        NB = set()

        # find the nearest and next nearest neighbors of u and update tv for v in Γ(u)
        for v in my_predecessor_function(u):
            NB.add(v)
            t[v] += 1
            for w in my_predecessor_function(v):
                if w not in S:
                    NB.add(w)

        # update gddv for all v in NB
        for v in NB:
            sumtw = 0
            for w in my_predecessor_function(v):
                if w not in S:
                    sumtw += t[w]

            dv = my_degree_function(v)

            if model == 'LT':
                GDD[v] = dv - p_values[v] * sumtw
            elif model == 'WC':
                GDD[v] = dv - (p_values[v] * (dv - t[v]) + 0.5 * p_values[v] * t[v] * (t[v] - 1))
            else:  # default to independent cascade
                GDD[v] = dv - 2 * t[v] - (dv - t[v]) * t[v] * p_values[v] + 0.5 * t[v] * (t[v] - 1) * p_values[v] - sumtw * p_values[v]

            if GDD[v] < 0:
                GDD[v] = 0

    return S

def general_greedy(k, G, p, no_simulations, model, max_time, random_generator):
	S = []

	for i in range(k):
		maxinfl_i = (-1, -1)
		v_i = -1
		for v in list(set(G.nodes()) - set(S)):
			eval_tuple =  SNSim.evaluate(G, S+[v], p, no_simulations, model, max_time)
			if eval_tuple[0] > maxinfl_i[0]:
				maxinfl_i = (eval_tuple[0], eval_tuple[1])
				v_i = v

		S.append(v_i)
		print(i+1, maxinfl_i[0], maxinfl_i[1], S)

	return S

def CELF(k, G, p, no_simulations, model, max_time, random_generator):
	A = []
	max_delta = len(G.nodes()) + 1
	delta = {}
	for v in G.nodes():
		delta[v] = max_delta
	curr = {}
	T = 0
	while len(A) < k:
		print("A: ", A, 'k: ', k)
		for j in set(G.nodes()) - set(A):
			curr[j] = False
		while True:
			# find the node s from V-A which maximizes delta[s]
			max_curr = -1
			s = -1
			for j in set(G.nodes()) - set(A):
				if delta[j] > max_curr:
					max_curr = delta[j]
					s = j
			# evaluate s only if curr = False
			if curr[s]:
				A.append(s)
				# the result for this seed set is:
				res = SNSim.evaluate(G, A, p, no_simulations, model, max_time, random_generator=random_generator)
				T += res[2]  				
				break
			else:
				eval_after  = SNSim.evaluate(G, A+[s], p, no_simulations, model, max_time,random_generator=random_generator)
				T += eval_after[2]  
				eval_before = SNSim.evaluate(G, A, p, no_simulations, model, max_time, random_generator=random_generator)
				T += eval_before[2] 
				delta[s] = eval_after[0] - eval_before[0]
				curr[s] = True

	return A , T

def single_discount_high_degree_nodes(k, G):
	if nx.is_directed(G):
		my_predecessor_function = G.predecessors
		my_degree_function = G.out_degree
	else:
		my_predecessor_function = G.neighbors
		my_degree_function = G.degree

	S = []
	ND = {}
	for n in G.nodes():
		ND[n] = my_degree_function(n)

	for i in range(k):
		# find the node of max degree not already in S
		u = max(set(list(ND.keys())) - set(S), key=(lambda key: ND[key]))
		S.append(u)

		# discount out-edges to u from all other nodes
		for v in my_predecessor_function(u):
			ND[v] -= 1

	return S

####################################################################################
def read_arguments():
    """
	Algorithm arguments, it is sufficient to specify all the parameters in the
	.json config file, which should be given as a parameter to the script, it
	should contain all the other script parameters.
	"""
    parser = argparse.ArgumentParser(
        description='Evolutionary algorithm computation.'
    )
    # Problem setup.
    parser.add_argument('--method', default="CELF", choices=['CELF', 'GDD', 'SDD'],
                        help='heuritics')
    parser.add_argument('--model', default="WC", choices=['LT', 'WC', 'IC'],
                        help='Influence propagation model.')

    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    args_base = read_arguments()

    choices=['email_di', 'facebook_combined_un', 'gnutella_di', 'lastfm_un','wiki-vote_di','CA-HepTh_un']


    args = {}
    args["k"] = 100
    args["p"] = 0.05
    args["no_simulations"] = 100
    args["model"] = args_base["model"]
    D = {}
    D_N = {}
    args["heuristic"] = args_base["method"]
    for c in choices:
        args["graph"] = c
        
        V, VV = {}, {}
        for seed_random in range(10):
            if f'heuristics/{args["heuristic"]}-{args["model"]}-{args["graph"]}_spread.csv ' in os.listdir('heuristics/'): continue
            args["random_generator"] = random.Random(seed_random)
            args["graph"] = c
            args["max_time"] = 5
            G, graph_name, args["communities"] = get_graph(args)

            args["bound_communities"] = [1/len(args["communities"]) for item in args["communities"]]  
            low_bound = [0 for item in args["communities"]]
            low_bound[0] = 1
            args["lower_bound_communities"] =   spatial.distance.jensenshannon(args["bound_communities"] ,  low_bound , base=2)
            #exit(0)
            if args["model"] == 'WC':
                if nx.is_directed(G):
                    p_values = {node: 1/G.in_degree(node) if G.in_degree(node) != 0 else 1  for node in G.nodes()}
                else:
                    p_values = {node: 1/G.degree(node) for node in G.nodes()}
            elif args["model"] == 'LT':
                if nx.is_directed(G):
                    p_values = {node: 1/G.in_degree(node) if G.in_degree(node) != 0 else 1  for node in G.nodes()}
                else:
                    p_values = {node: 1/G.degree(node) for node in G.nodes()}
            elif args["model"] == 'IC':
                p_values = {node: args["p"] for node in G.nodes()}

                
            if args["heuristic"] == 'GDD':
                A = generalized_degree_discount(args["k"], G, p_values, args["model"], args["random_generator"])
            elif args["heuristic"] == 'GG':
                A = general_greedy(args["k"], G, args["p"], args["no_simulations"], args["model"], args["max_time"], args["random_generator"] )
            elif args["heuristic"] == 'CELF':
                A, B = CELF(args["k"], G, args["p"], args["no_simulations"], args["model"], args["max_time"], args["random_generator"] )
            elif args["heuristic"] == 'SDD':
                A = single_discount_high_degree_nodes(args["k"], G)
            solutions = []
            for i in range(len(A)):
                item = A[:i+1]
                fairness = get_fairness(args, item)
                args["max_budget"] = max_budget(G,args["k"])
                if nx.is_directed(G):
                    b = sum([G.out_degree(i) for i in list(item)])
                elif nx.is_directed(G) == False:
                    b = sum([G.degree(i) for i in list(item)])
                influence_mean, _, comm, time_ = MonteCarlo_simulation(G, item,args["p"] ,p_values, args["no_simulations"], args["model"], args["communities"], args["bound_communities"],  args["max_time"], args["lower_bound_communities"], args["random_generator"])
                influence = influence_mean/G.number_of_nodes()
                seed = (args["k"]+1-len(item))/args["k"]
                nodes = len(item)/G.number_of_nodes()
                comm = 1 - comm
                fairness = 1- fairness
                time_ = 1- time_/args["max_time"]
                b = 1- b/args["max_budget"]
                solutions.append(([seed, influence, comm, fairness,  b, time_]))
                print('Seed', seed, 'Influence', influence, 'Fairness', fairness, 'Time', time_, 'Budget', b, 'Community', comm)
                V[f'{args["heuristic"]}-{args["model"]}-{args["graph"]}-{seed_random}-{i}'] = [args["heuristic"], args["model"], args["graph"], seed_random, i, influence,seed, nodes,comm, fairness, b, time_]
                spread_seed = [x[:2] for x in solutions]
                spread_seed_comm = [spread_seed[i] + [x[2]] for i,x in enumerate(solutions)]
                spread_seed_fairness = [spread_seed[i] + [x[3]] for i,x in enumerate(solutions)]
                spread_seed_budget = [spread_seed[i] + [x[4]] for i,x in enumerate(solutions)]
                spread_seed_time= [spread_seed[i] + [x[5]] for i,x in enumerate(solutions)]


                hv_seed_spread = hypervolume_seed_spread(spread_seed, args)
                hv_seed_spread_comm = hypervolume_with_one(spread_seed_comm, args)
                hv_seed_spread_fairness = hypervolume_with_one(spread_seed_fairness, args)
                hv_seed_spread_budget = hypervolume_with_one(spread_seed_budget, args)
                hv_seed_spread_time = hypervolume_with_one(spread_seed_time, args)
                VV[f'{args["heuristic"]}-{args["model"]}-{args["graph"]}-{seed_random}-{i}'] = [args["heuristic"], args["model"], args["graph"], seed_random, i, hv_seed_spread, hv_seed_spread_comm, hv_seed_spread_fairness, hv_seed_spread_budget, hv_seed_spread_time]      



            df_ = pd.DataFrame.from_dict(V, orient='index', columns=['heuristic', 'model', 'graph', 'seed', 'k', 'influence', 'seed_set', 'n_nodes','community', 'fairness', 'budget', 'time'])   
            df_.to_csv(f'heuristics/{args["heuristic"]}-{args["model"]}-{args["graph"]}_spread.csv', index=False)

            df_ = pd.DataFrame.from_dict(VV, orient='index', columns=['heuristic', 'model', 'graph', 'seed', 'k', 'spread', 'community', 'fairness', 'budget', 'time'])
            df_.to_csv(f'heuristics/{args["heuristic"]}-{args["model"]}-{args["graph"]}_hv.csv', index=False)

            

            spread_seed = [x[:2] for x in solutions]
            spread_seed_comm = [spread_seed[i] + [x[2]] for i,x in enumerate(solutions)]
            spread_seed_fairness = [spread_seed[i] + [x[3]] for i,x in enumerate(solutions)]
            spread_seed_budget = [spread_seed[i] + [x[4]] for i,x in enumerate(solutions)]
            spread_seed_time= [spread_seed[i] + [x[5]] for i,x in enumerate(solutions)]


            hv_seed_spread = hypervolume_seed_spread(spread_seed, args)
            hv_seed_spread_comm = hypervolume_with_one(spread_seed_comm, args)
            hv_seed_spread_fairness = hypervolume_with_one(spread_seed_fairness, args)
            hv_seed_spread_budget = hypervolume_with_one(spread_seed_budget, args)
            hv_seed_spread_time = hypervolume_with_one(spread_seed_time, args)

            pr = [hv_seed_spread/hv_seed_spread, hv_seed_spread_comm/hv_seed_spread, hv_seed_spread_fairness/hv_seed_spread, hv_seed_spread_budget/hv_seed_spread, hv_seed_spread_time/hv_seed_spread]
            pr = [round(x,3) for x in pr]

            normal = [hv_seed_spread, hv_seed_spread_comm, hv_seed_spread_fairness, hv_seed_spread_budget, hv_seed_spread_time]
            normal = [round(x,3) for x in normal]
            print('Base HV', hv_seed_spread, 'others', pr)
            print('Base HV Normal', normal)
            D[f'{args["heuristic"]}-{args["model"]}-{args["graph"]}-{seed_random}'] = [args["heuristic"], args["model"], args["graph"],  seed_random, hv_seed_spread, hv_seed_spread_comm, hv_seed_spread_fairness, hv_seed_spread_budget, hv_seed_spread_time]      
            D_N[f'{args["heuristic"]}-{args["model"]}-{args["graph"]}-{seed_random}'] = [args["heuristic"], args["model"], args["graph"], seed_random,hv_seed_spread/hv_seed_spread, hv_seed_spread_comm/hv_seed_spread, hv_seed_spread_fairness/hv_seed_spread, hv_seed_spread_budget/hv_seed_spread, hv_seed_spread_time/hv_seed_spread]



            df = pd.DataFrame.from_dict(D, orient='index', columns=['heuristic', 'model', 'graph',  'seed','spread', 'community', 'fairness', 'budget', 'time'])
            df.to_csv(f'general_spread-{args["heuristic"]}-{args["model"]}.csv', index=False)


            df = pd.DataFrame.from_dict(D_N, orient='index', columns=['heuristic', 'model', 'graph', 'seed','spread', 'community', 'fairness', 'budget', 'time'])
            df.to_csv(f'general_hv-{args["heuristic"]}-{args["model"]}.csv', index=False)
