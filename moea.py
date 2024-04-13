"""Evolutionary Algorithm"""

"""The functions in this script run Evolutionary Algorithms for influence maximization. Ideally, it will eventually contain both the single-objective (maximize influence with a fixed amount of seed nodes) and multi-objective (maximize influence, minimize number of seed nodes) versions. This relies upon the inspyred Python library for evolutionary algorithms."""
import random
import logging
from time import time
# local libraries
from src.spread.monte_carlo import MonteCarlo_simulation
from src.spread.monte_carlo_max_hop import MonteCarlo_simulation_max_hop as MonteCarlo_simulation_max_hop
from src.utils import *
from src.ea.observer import hypervolume_observer
from src.ea.evaluator import nsga2_evaluator
from src.ea.generator import nsga2_generator
from src.ea.crossover import ea_one_point_crossover, nsga2_super_operator, nsga2_super_operator_base, nsga2_crossover

from src.ea.terminators import generation_termination
from src.ea.mutators import *

# inspyred libriaries
import inspyred
from inspyred.ec import *
from scipy import spatial
import networkx as nx

def max_budget(G,k):
    if nx.is_directed(G):
        b = sorted([G.out_degree(i) for i in G.nodes()], reverse=True)
    elif nx.is_directed(G) == False:
        b = sorted([G.degree(i) for i in G.nodes()], reverse=True)
    b = sum(b[:k])

    return b

"""
Multi-objective evolutionary influence maximization. Parameters:
    G: networkx graph
    p: probability of influence spread 
    no_simulations: number of simulations
    model: type of influence propagation model
    population_size: population of the EA (default: value)
    offspring_size: offspring of the EA (default: value)
    max_generations: maximum generations (default: value)
    min_seed_nodes: minimum number of nodes in a seed set (default: 1)
    max_seed_nodes: maximum number of nodes in a seed set (default: 1% of the graph size)
    n_threads: number of threads to be used for concurrent evaluations (default: 1)
    random_gen: already initialized pseudo-random number generation
    initial_population: individuals (seed sets) to be added to the initial population (the rest will be randomly generated)
    population_file: name of the file that will be used to store the population at each generation (default: file named with date and time)
    no_obj: number of objcetive function in the multi-objcetive optimization
    """
def moea_influence_maximization(G,args,fitness_function_kargs=dict(),random_gen=random.Random(),population_file=None, initial_population=None,m=None) :
    # initialize multi-objective evolutionary algorithm, NSGA-II
    nodes = list(G.nodes)
    if args["k"] == None: 
        max_seed_nodes = int(0.1 * len(nodes))
        logging.debug("Maximum size for the seed set has been set to %d" % max_seed_nodes)
    if args["t"] == None:
        args["t"] = np.inf
    if population_file == None :
        population_file = "RandomGraph-N{nodes}-E{edges}-population.csv".format(nodes=len(G.nodes), edges=G.number_of_edges())
    
    fitness_function = MonteCarlo_simulation
    fitness_function_kargs["random_generator"] = random_gen # pointer to pseudo-random number generator
    m = [1/len(args["communities"]) for item in args["communities"]]    
    m_ = [0 for item in args["communities"]]
    m_[0] = 1
    lower_bound_comm = spatial.distance.jensenshannon(m, m_ , base=2)
    ea = inspyred.ec.emo.NSGA2(random_gen)

    if args["version"] == 'graph-aware':
        ea.variator = [ea_one_point_crossover,nsga2_super_operator]
    elif args["version"] == 'base':
        ea.variator = [nsga2_crossover,nsga2_super_operator_base]

    ea.terminator = [generation_termination]
    ea.observer = [hypervolume_observer]
    
    bounder = inspyred.ec.DiscreteBounder(nodes)
    if nx.is_directed(G):
        degree_list = {node: 1 / G.in_degree(node) if G.in_degree(node) != 0 else 1 for node in G.nodes()}
    elif nx.is_directed(G) == False:
        degree_list = {node: 1 / G.degree(node) for node in G.nodes()}

    # start the evolutionary process
    ea.evolve(
        generator = nsga2_generator,
        evaluator = nsga2_evaluator,
        bounder= bounder,
        maximize = True,
        seeds = initial_population,
        pop_size = args["population_size"],
        num_selected = args["offspring_size"],
        generations_budget=args["max_generations"],
        #max_generations=int(max_generations*0.9), #no termination criteria used in this work
        tournament_size=args["tournament_size"],
        num_elites=args["num_elites"],

        # all arguments below will go inside the dictionary 'args'
        G = G,
        p = args["p"],
        model = args["model"],
        no_simulations = args["no_simulations"],
        nodes = nodes,
        n_threads = 30,
        min_seed_nodes = 1,
        max_seed_nodes = args["k"],
        population_file = population_file,
        time_previous_generation = time(), # this will be updated in the observer
        fitness_function = fitness_function,
        fitness_function_kargs = fitness_function_kargs,
        mutation_operator=ea_global_random_mutation,
        time_max = args["t"],

        #others IM
        graph = G,
        hypervolume = [], # keep track of HV trend throughout the generations
        obj_functions = args["obj_functions"], 
        lower_bound_comm = lower_bound_comm,
        objective_trend = {}, # keep track of Time (Activation Attempts) trend throughout the generations
        communities = args["communities"],
        population_trend = [],
        archiver_trend = [], 
        max_degree_budget = max_budget(G,args["k"]),
        bound_communities = m,
        hypervolume_trend = [],
        degrees_graph = degree_list

    )

    all = ea._kwargs['objective_trend']
    m = int(max_budget(G,args["k"]))

    if args["obj_functions"]  == ['spread', 'seed']:
        seed_sets_normalized = [[individual.candidate, individual.fitness[0], individual.fitness[1]] for individual in ea.archive] 
        seed_sets_default = [[individual.candidate, individual.fitness[0] * G.number_of_nodes(), args["k"]*(1-individual.fitness[1])+1] for individual in ea.archive] 
        to_csv2(seed_sets_normalized, seed_sets_default, population_file, G.number_of_nodes())
        plot_all_trend(all, ea.archive, G.number_of_nodes(), args["k"], m,args["t"],population_file)
    elif args["obj_functions"] == ['spread', 'seed', 'time']:
        seed_sets_normalized = [[individual.candidate, individual.fitness[0], individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        seed_sets_default = [[individual.candidate, individual.fitness[0] * G.number_of_nodes(), args["k"]*(1-individual.fitness[1])+1, args["t"]* (1-individual.fitness[2])] for individual in ea.archive] 
        to_csv(seed_sets_normalized, seed_sets_default, population_file)
        plot_all_trend(all, ea.archive, G.number_of_nodes(), args["k"], m,args["t"],population_file)
    elif args["obj_functions"] == ['spread', 'seed', 'communities'] or  args["obj_functions"] == ['spread', 'seed', 'fairness']:
        seed_sets_normalized = [[individual.candidate, individual.fitness[0], individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        seed_sets_default = [[individual.candidate, individual.fitness[0] * G.number_of_nodes(), args["k"]*(1-individual.fitness[1])+1, 1 - individual.fitness[2]] for individual in ea.archive] 
        to_csv(seed_sets_normalized, seed_sets_default, population_file)
        plot_all_trend(all, ea.archive, G.number_of_nodes(), args["k"], m,args["t"],population_file)
    elif  args["obj_functions"] == ['spread', 'seed', 'budget']:
        seed_sets_normalized = [[individual.candidate, individual.fitness[0], individual.fitness[1], individual.fitness[2]] for individual in ea.archive] 
        seed_sets_default = [[individual.candidate, individual.fitness[0] * G.number_of_nodes(), args["k"]*(1-individual.fitness[1])+1, int(max_budget(G,args["k"])*(1 - individual.fitness[2]))] for individual in ea.archive] 
        to_csv(seed_sets_normalized, seed_sets_default, population_file)
        plot_all_trend(all, ea.archive, G.number_of_nodes(), args["k"], m,args["t"],population_file)
    elif args["obj_functions"] == ['spread', 'seed', 'communities', 'fairness', 'budget', 'time']:
        seed_sets_normalized = [[individual.candidate, individual.fitness[0], individual.fitness[1], individual.fitness[2], individual.fitness[3], individual.fitness[4], individual.fitness[5]] for individual in ea.archive] 
        seed_sets_default = [[individual.candidate, individual.fitness[0] * G.number_of_nodes(), args["k"]*(1-individual.fitness[1])+1, 1 - individual.fitness[2],1 - individual.fitness[3], int(max_budget(G,args["k"])*(1 - individual.fitness[4])), args["t"]* (1-individual.fitness[5])] for individual in ea.archive] 
        to_csv(seed_sets_normalized, seed_sets_default, population_file)
        plot_all_trend(all, ea.archive, G.number_of_nodes(), args["k"], m,args["t"],population_file)

    #print number of nodes of the graph
    logging.info("Number of nodes: %d" % len(G.nodes))
    return True


