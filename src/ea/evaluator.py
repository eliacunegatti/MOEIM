from src.spread.monte_carlo import MonteCarlo_simulation
import inspyred
import numpy as np
from scipy import spatial
from tqdm import tqdm
from joblib import Parallel, delayed
from copy import deepcopy
import networkx as nx

def get_fairness(args, A_set):
    check = []
    for item in args["communities"]:
        intersection = set.intersection(set(item),set(A_set))
        check.append(len(set(intersection))/len(set(item)))

    fairness = 0
    if sum(check) > 0:
        comm_ratio = np.array(check) / sum(np.array(check))
        fairness = spatial.distance.jensenshannon(comm_ratio, args["bound_communities"])/args["lower_bound_comm"]
    else:
        fairness = 1

    return fairness


def nsga2_evaluator(candidates, args):
    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]
    fitness_function = args["fitness_function"]
    fitness_function_kargs = args["fitness_function_kargs"]
    k = args["max_seed_nodes"]
    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # depending on how many threads we have at our disposal,
    # we use a different methodology
    # if we just have one thread, let's just evaluate individuals old style 

    #calculate Time (Activation Attempts) for every individual in the population 
    time_gen = [None] * len(candidates)
    if n_threads == 1 :
        #for loop over all individuals with enumerate and tqdm  for progress bar
        for index, A in tqdm(enumerate(candidates), total=len(candidates), desc=f"Processing"):
            A_set = set(A)
            fitness_function_args = [G, A_set, args["p"], args["degrees_graph"], no_simulations, model, args["communities"], args["bound_communities"], args["time_max"], args["lower_bound_comm"]]
            if nx.is_directed(G):
                b = sum([G.out_degree(i) for i in list(A_set)])
            elif nx.is_directed(G) == False:
                b = sum([G.degree(i) for i in list(A_set)])
            fairness = get_fairness(args, A_set)

            influence_mean, _, comm,  time = fitness_function(*fitness_function_args, **fitness_function_kargs)
            if str(sorted(list(A_set), reverse=True)) in args["objective_trend"].keys():
                if (influence_mean / G.number_of_nodes())  > args["objective_trend"][str(sorted(list(A_set), reverse=True))][0]:
                    args["objective_trend"][str(sorted(list(A_set), reverse=True))] = [(influence_mean / G.number_of_nodes()) , (k+1-len(A_set))/k , 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1-time/args["time_max"]]
                else:
                    pass
            else:
                args["objective_trend"][str(sorted(list(A_set), reverse=True))] = [(influence_mean / G.number_of_nodes()) , (k+1-len(A_set))/k , 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1 - time/args["time_max"]]

            if args["obj_functions"] == ['spread','seed']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k )])
            if args["obj_functions"] == ['spread','seed','budget']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1- (b/args["max_degree_budget"])])
            elif args["obj_functions"] == ['spread','seed','time']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1 - time/args["time_max"]])
            elif args["obj_functions"] == ['spread','seed','communities']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm])
            elif args["obj_functions"] == ['spread','seed','communities', 'time']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm, 1 - time/args["time_max"]])
            elif  args["obj_functions"] == ['spread','seed','fairness']:  
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ),1-fairness])     
            elif args["obj_functions"] == ['spread', 'seed', 'communities', 'fairness', 'budget', 'time']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1- time/args["time_max"]])    
    else :
        parallel_eval_args = []
        for index, A in (enumerate(candidates)):
            A_set = set(A)

            fitness_function_args = [G, A_set,  args["p"], args["degrees_graph"], no_simulations, model, args["communities"], args["bound_communities"], args["time_max"],args["lower_bound_comm"]]
            parallel_eval_args.append(fitness_function_args)
        with Parallel(n_threads) as parallel:
            outputs = parallel(delayed(fitness_function)(*fitness_function_args,**deepcopy(fitness_function_kargs)) for fitness_function_args in tqdm(parallel_eval_args))

        for index, A in tqdm(enumerate(candidates), total=len(candidates), desc=f"Processing"):
            A_set = set(A)
            if nx.is_directed(G):
                b = sum([G.out_degree(i) for i in list(A_set)])
            elif nx.is_directed(G) == False:
                b = sum([G.degree(i) for i in list(A_set)])
            fairness = get_fairness(args, A_set)
            influence_mean, _, comm,  time = outputs[index]
            if str(sorted(list(A_set), reverse=True)) in args["objective_trend"].keys():
                if (influence_mean / G.number_of_nodes())  > args["objective_trend"][str(sorted(list(A_set), reverse=True))][0]:
                    args["objective_trend"][str(sorted(list(A_set), reverse=True))] = [(influence_mean / G.number_of_nodes()) , (k+1-len(A_set))/k , 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1-time/args["time_max"]]
                else:
                    pass
            else:
                args["objective_trend"][str(sorted(list(A_set), reverse=True))] = [(influence_mean / G.number_of_nodes()) , (k+1-len(A_set))/k , 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1- time/args["time_max"]]

            if args["obj_functions"] == ['spread','seed']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k )])
            if args["obj_functions"] == ['spread','seed','budget']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1- (b/args["max_degree_budget"])])
            elif args["obj_functions"] == ['spread','seed','time']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1- time/args["time_max"]])
            elif args["obj_functions"] == ['spread','seed','communities']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm])
            elif args["obj_functions"] == ['spread','seed','communities', 'time']:
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm, 1- time/args["time_max"]])
            elif  args["obj_functions"] == ['spread','seed','fairness']:  
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ),1-fairness])
            elif args["obj_functions"] == ['spread', 'seed', 'communities', 'fairness', 'budget', 'time']: 
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()) , ((k+1-len(A_set))/k ), 1-comm, 1-fairness, 1- (b/args["max_degree_budget"]), 1 - time/args["time_max"]])    
    return fitness


"""
from concurrent.futures import ThreadPoolExecutor
import inspyred
import numpy as np
from scipy import spatial
from tqdm import tqdm

# ... (other imports and functions)

def nsga2_evaluator(candidates, args):
    n_threads = args["n_threads"]
    G = args["G"]
    p = args["p"]
    model = args["model"]
    no_simulations = args["no_simulations"]
    fitness_function = args["fitness_function"]
    fitness_function_kargs = args["fitness_function_kargs"]
    k = args["max_seed_nodes"]
    # we start with a list where every element is None
    fitness = [None] * len(candidates)

    # Function to evaluate an individual in parallel
    def evaluate_individual(index, A):
        A_set = set(A)
        fitness_function_args = [G, A_set, args["degrees_graph"], no_simulations, model, args["communities"], args["bound_communities"], args["time_max"]]
        b = sum([G.degree(i) for i in list(A_set)])
        fairness = get_fairness(args, A_set)

        influence_mean, _, comm, time = fitness_function(*fitness_function_args, **fitness_function_kargs)

        # ... (rest of the evaluation logic, including updating args["objective_trend"])

        return index, influence_mean, comm, time

    # Depending on the number of threads, use a different methodology
    if n_threads == 1:
        for index, A in tqdm(enumerate(candidates), total=len(candidates), desc="Processing"):
            index, influence_mean, comm, time = evaluate_individual(index, A)
            # Update fitness based on the individual's evaluation
            fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()), ...])  # Update this line with your fitness calculation.

    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Use executor to evaluate individuals in parallel
            futures = [executor.submit(evaluate_individual, index, A) for index, A in enumerate(candidates)]

            for future in tqdm(inspyred.ec.Evaluator.parallel_wait(futures), total=len(futures), desc="Processing"):
                index, influence_mean, comm, time = future.result()
                # Update fitness based on the individual's evaluation
                fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / G.number_of_nodes()), ...])  # Update this line with your fitness calculation.

    return fitness


"""