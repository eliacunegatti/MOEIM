import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial

# local libraries
from src.load import read_graph, read_graph_total
from src.utils import inverse_ncr
from moea import moea_influence_maximization
# spread models

from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
# smart initialization
from src.smart_initialization import degree_random
from src.nodes_filtering.select_best_spread_nodes import filter_best_nodes as filter_best_spread_nodes
from src.nodes_filtering.select_min_degree_nodes import filter_best_nodes as filter_min_degree_nodes


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
    parser.add_argument('--k', type=float, default=0.2, help='Seed set size as percentage of the whole network.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of influence spread in the IC model.')
    parser.add_argument('--no_simulations', type=int, default=100,
                        help='Number of simulations for spread calculation'
                             ' when the Monte Carlo fitness function is used.')
    parser.add_argument('--model', default="WC", choices=['LT', 'WC', 'IC'],
                        help='Influence propagation model.')

    parser.add_argument('--obj_functions', default=['spread', 'seed'], type=str, 
                        choices= [['spread', 'seed'],['spread', 'seed', 'time'], ['spread', 'seed', 'communities'], ['spread', 'seed','budget'],['spread', 'seed', 'fairness']],
                        help='Objective functions to be optimized.')
    # EA setup.
    parser.add_argument('--population_size', type=int, default=100,
                        help='EA population size.')
    parser.add_argument('--offspring_size', type=int, default=100,
                        help='EA offspring size.')
    parser.add_argument('--max_generations', type=int, default=100,
                        help='Maximum number of generations.')
    parser.add_argument('--tournament_size', type=int, default=5,
                        help='EA tournament size.')
    parser.add_argument('--num_elites', type=int, default=2,
                        help='EA number of elite individuals.')
    parser.add_argument('--no_runs', type=int, default=10,
                        help='Number of runs of the EA.')
    parser.add_argument('--version', default="graph-aware", choices=['graph-aware', 'base'],
                        help='Smart Initialization and graph-aware operators or no Smart Initialization and base mutator operators.')
    
    parser.add_argument('--experimental_setup', default="setting1", choices=['setting1', 'setting2'],
                        help='Setting 1 and 2 of the experimental section.')
    # EA improvements setup.
    # Smart initialization.
    parser.add_argument('--smart_initialization_percentage', type=float,
                        default=0.33,
                        help='Percentage of "smart" initial population, to be '
                             'specified when multiple individuals technique is '
                             'used.')
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

    # Graph setup.
    parser.add_argument('--graph', default='jazz_un',
                        choices=['facebook_combined_un', 'email_di', 'soc-epinions_di', 'gnutella_di', 'wiki-vote_di','CA-HepTh_un', 'lastfm_un','power_grid_un', 'jazz_un', 'cora-ml_un'],
                        help='Dataset name (un_ for undirected and di_ for directed).')

    # Input/output setup.
    parser.add_argument('--out_dir', 
                        default='experiments/', 
                        type=str,
                        help='Location of the output directory in case if '
                             'outfile is preferred to have a default name.')
    

    args = parser.parse_args()
    args = vars(args)
    return args

#------------------------------------------------------------------------------------------------------------#  

#SMART INITIALIZATION
def create_initial_population(G, args, prng=None, nodes=None):
    """
	Smart initialization techniques.
	"""
    # smart initialization
    initial_population = None
    initial_population = degree_random(args["k"], G,
                                        int(args["population_size"] * args["smart_initialization_percentage"]),
                                        prng, nodes=nodes)

    return initial_population
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

    filename = 'data/graphs/graphs_cleaned/{0}.txt'.format(args["graph"])
    graph_name = str(os.path.basename(filename))
    graph_name = graph_name.replace(".txt", "")

    G = read_graph_total(filename)
    return G, graph_name



def create_folder(args, graph_name):
    if args["out_dir"] != None:
        path = '{0}{1}-{2}-{3}-{4}-{5}'.format(args["out_dir"],graph_name,args["model"], args["obj_functions"], args["version"], args["experimental_setup"])
    else:
        path = '{0}-{1}-{2}-{3}-{4}'.format(graph_name,args["model"], args["obj_functions"], args["version"], args["experimental_setup"])
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    return path

def get_filter_nodes(args, G):
    my_degree_function = G.degree
    mean = []
    for item in G:
        mean.append(my_degree_function[item])
        
    #define minimum degree threshold as the average degree +1 
    args["min_degree"] = int(np.mean(mean)) + 1
    nodes_filtered = filter_nodes(G, args)
    return nodes_filtered



def get_nodes_filtered(args):
    df = pd.read_csv(f'data/graphs/smart_init.csv', header=None)

    #del first row
    df = df.iloc[1:]
    print(df)
    #iterate the dataframe by row
    for index, row in df.iterrows():
        print(row[0])
        if args["graph"] in row[0] and args["model"] in row[0]:
            nodes_filtered = row[1:]
            break
    #remove all nan values from nodes_filtered
    nodes_filtered = [int(x) for x in nodes_filtered if str(x) != 'nan']
    return nodes_filtered
#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    args = read_arguments()
    G, graph_name = get_graph(args)
    df = pd.read_csv(f'data/graphs/graph_communities/{args["graph"]}.csv', sep=',')
    args["communities"]  = [[] for i in range(len(set(df["comm"].to_list())))]
    nodes_ = df["node"].to_list()
    comm_ = df["comm"].to_list()
    print(len(args['communities']))
    for i in range(len(nodes_)):
        args['communities'][comm_[i]-1].append(nodes_[i])


    if args["experimental_setup"] == 'setting1':
        args["k"] = 100
        args["t"] = 5
        print('k: ', args["k"]) 
        choices= [['spread', 'seed'],['spread', 'seed', 'time'],['spread', 'seed', 'budget'],['spread', 'seed','communities'],['spread', 'seed', 'fairness'],['spread', 'seed', 'communities', 'fairness', 'budget', 'time']]
    elif args["experimental_setup"] == 'setting2':
        args["k"] = int(0.2 * G.number_of_nodes())
        args["t"] = None
        print('k: ', args["k"]) 

        choices= [['spread', 'seed']]

    for setting in choices:
        args["obj_functions"] = setting
        path = create_folder(args, graph_name)        

        for run in range(args["no_runs"]):
            prng = random.Random(run)
            if args["version"] == 'graph-based':
                nodes_filtered = get_nodes_filtered(args)
                initial_population = create_initial_population(G, args, prng, nodes_filtered)
            else:
                initial_population = None

            #Print Graph's information and properties
            #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            #logging.info(nx.classes.function.info(G))
            
            
            file_path = 'run-{0}'.format(run+1)
            file_path = path+'/'+file_path       

            file_to_check = file_path + '-population.csv'
            if os.path.isfile(file_to_check): continue
            ##MOEA INFLUENCE MAXIMIZATION WITH FITNESS FUNCTION MONTECARLO_SIMULATION
            seed_sets = moea_influence_maximization(G, args, random_gen=prng,population_file=file_path,initial_population=initial_population)

