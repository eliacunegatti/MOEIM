import sys
sys.path.insert(0, '')
import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial

# local libraries
from src.load import read_graph
from src.utils import inverse_ncr
from moea import moea_influence_maximization
# spread models
from src.spread.monte_carlo_2_obj import MonteCarlo_simulation as MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
# smart initialization
from src.smart_initialization import degree_random
from src.nodes_filtering.select_best_spread_nodes import filter_best_nodes as filter_best_spread_nodes
from src.nodes_filtering.select_min_degree_nodes import filter_best_nodes as filter_min_degree_nodes

#code based on https://github.com/katerynak/Influence-Maximization

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
    parser.add_argument('--k', type=float, default=0.05, help='Seed set size as percentage of the whole network.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of influence spread in the IC model.')
    parser.add_argument('--no_simulations', type=int, default=100,
                        help='Number of simulations for spread calculation'
                             ' when the Monte Carlo fitness function is used.')

    # Smart initialization.
    parser.add_argument('--smart_initialization', default="degree_random",
                        choices=["none", "degree", "eigenvector", "katz",
                                 "closeness", "betweenness",
                                 "community", "community_degree",
                                 "community_degree_spectral", "degree_random",
                                 "degree_random_ranked"],
                        help='If set, an individual containing the best nodes '
                             'according to the selected centrality metric will '
                             'be inserted into the initial population.')
    parser.add_argument('--smart_initialization_percentage', type=float,
                        default=0.5,
                        help='Percentage of "smart" initial population, to be '
                             'specified when multiple individuals technique is '
                             'used.')
    parser.add_argument("--filter_best_spread_nodes", type=str, nargs="?",
                        const=True, default=True,
                        help="If true, best spread filter is used.")
    parser.add_argument("--search_space_size_min",
                        type=float,
                        #default=None,
                        default=1e9,
                        help="Lower bound on the number of combinations.")
    parser.add_argument("--search_space_size_max",
                        type=float,
                        default=1e11,
                        #default=None,
                        help="Upper bound on the number of combinations.")

    parser.add_argument('--random_seed', type=int, default=random.randint(1,10000),
                        help='Seed to initialize the pseudo-random number '
                             'generation.')
    args = parser.parse_args()
    args = vars(args)
    return args

#------------------------------------------------------------------------------------------------------------#  


def filter_nodes(G, args):
    """
	Selects the most promising nodes from the graph	according to specified
	input arguments techniques.

	:param G:
	:param args:
	:return:
	"""
    # nodes filtering
    nodes = None
    if args["filter_best_spread_nodes"]:
        best_nodes = inverse_ncr(args["search_space_size_min"], args["k"])
        error = (inverse_ncr(args["search_space_size_max"], args["k"]) - best_nodes) / best_nodes
        filter_function = partial(MonteCarlo_simulation_max_hop, G=G, random_generator=prng, p=args["p"], model=args["model"],
                                  max_hop=3, no_simulations=1)
        nodes = filter_best_spread_nodes(G, best_nodes, error, filter_function)
    nodes = filter_min_degree_nodes(G, args["min_degree"], nodes)

    return nodes

#------------------------------------------------------------------------------------------------------------#

#LOCAL FUNCTIONS
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

    return G, graph_name


def get_filter_nodes(args, G):
    my_degree_function = G.degree
    mean = []
    for item in G:
        mean.append(my_degree_function[item])
        
    #define minimum degree threshold as the average degree +1 
    args["min_degree"] = int(np.mean(mean)) + 1

    nodes_filtered = filter_nodes(G, args)
    return nodes_filtered

#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    choices=['jazz_un','CA-HepTh_un','cora-ml_un', 'email_di', 'facebook_combined_un', 'gnutella_di', 'lasfm_un','power_grid_un', 'soc-epinions_di', 'wiki_vote_di']
    model = ['IC', 'WC', 'LT'] 
    res = {}
    for c in choices:
        for m in model:
            args = read_arguments()
            print('loading graph: ', c)
            args["graph"] = c
            args["model"] = m
            G, graph_name = get_graph(args)
            

            print('Computing for graph: ', c, ' and model: ', m)

            prng = random.Random(args["random_seed"])

            #Calculate k based on network size
            #args["k"] = int(G.number_of_nodes() * args["k"])
            args["k"] = 100
            args["obj_functions"] = ['spread', 'seed']
                
            #select best nodes with smart initiliazation
            nodes_filtered = get_filter_nodes(args,G)
            res[f'{c}-{m}'] = [int(x) for x in nodes_filtered]
            del G, args,nodes_filtered
        #save results from dictionary to csv
        df = pd.DataFrame.from_dict(res, orient='index')
        print(df)
        df.to_csv('filtered_nodes.csv', index=True, header=True)



