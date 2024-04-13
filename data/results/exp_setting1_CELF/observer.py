import os
import logging
import numpy as np
import inspyred.ec
import pandas as pd
from pymoo.indicators.hv import Hypervolume
#local libraries
from src.ea.generators import generator_new_nodes
from src.utils import diversity, individuals_diversity
import copy
def minimize_obj(A):
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] = -float(A[i][j])
    return np.array(A)
	
#function for computing the hypervolume of two objective functions namely seed and spread
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


def get_hv(A, args, no_obj=2):
    F = minimize_obj(copy.deepcopy(A))
    #print(F[0].shape, F.shape)
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    if no_obj == 2: 
        metric = Hypervolume(ref_point= np.array([0,0]),
                        norm_ref_point=False,
                        zero_to_one=False)
    else:
        metric = Hypervolume(ref_point= np.array([0,0,0]),
						norm_ref_point=False,
						zero_to_one=False)
    hv = metric.do(F)
    return hv



def adjust_population_size(num_generations, population, args):
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness

	improvement = (current_best - prev_best) / prev_best

	if "improvement" in args.keys():
		args["improvement"].pop(0)
		args["improvement"].append(improvement)
	else:
		args["improvement"] = [0] * 3
	if len(population) < 100:
		if sum(args["improvement"]) == 0 and num_generations > 2:
			# new_individuals = min(int(1/div), 10)
			new_individuals = 1
			for _ in range(new_individuals):
				# candidate = args["_ec"].generator(args["prng"], args)
				candidate = generator_new_nodes(args["prng"], args)
				args["_ec"].population.append(inspyred.ec.Individual(candidate=candidate))
				args["_ec"].population[-1].fitness = args["fitness_function"](A=candidate)[0]
			args["num_selected"] = len(args["_ec"].population)

		elif len(population) > args["min_pop_size"] and improvement > 0:
			min_fit_ind = args["_ec"].population[0]
			for ind in args["_ec"].population:
				if ind.fitness < min_fit_ind.fitness:
					min_fit_ind = ind
			args["_ec"].population.remove(min_fit_ind)
			args["num_selected"] = len(args["_ec"].population)


def ea_observer0(population, num_generations, num_evaluations, args):
	"""
	adjusting some dynamic parameters of the dynamic algorithms
	"""
	# check for repetitions
	for ind in population:
		if len(set(ind.candidate)) != len(ind.candidate):
			raise NameError("Nodes repetition inside an individual")
	# exploration weight exponential decay
	if args["mab"] is not None:
		args["mab"].exploration_weight = 1 / (num_generations + 1) ** (3)
		logging.debug("Mab selections: {}".format(args["mab"].n_selections))
	logging.debug("Population size: {}".format(len(population)))

	if args["dynamic_population"]:
		adjust_population_size(num_generations, population, args)


def ea_observer1(population, num_generations, num_evaluations, args):
	"""
	debug info, printing to stdout some generational info
	"""
	# to access to evolutionary computation stuff
	div = diversity(population)
	logging.debug("generation {}: diversity {}".format(num_generations, div))
	ind_div = individuals_diversity(population)
	logging.debug("generation {}: individuals diversity {}".format(num_generations, ind_div))

	return


def ea_observer2(population, num_generations, num_evaluations, args):
	"""
	printing generational log to out files
	"""

	# write current state of the population to a file
	sf = args["statistics_file"]

	# compute some generation statistics
	generation_stats = {}
	prev_best = args["prev_population_best"]
	current_best = max(population).fitness
	if prev_best > 0:
		generation_stats["improvement"] = (current_best - prev_best) / prev_best
	else:
		generation_stats["improvement"] = 0
	args["prev_population_best"] = current_best

	sf.seek(sf.tell()-1, os.SEEK_SET)

	sf.write(",{},".format(diversity(population)))
	sf.write("{},".format(generation_stats["improvement"]))
	if args["mab"] is not None:
		sf.write("{}\n".format(args["mab"].n_selections))
	else:
		sf.write("\n")

	return



def obj_observer(population, num_generations, num_evaluations, args):
    
	#obj = str(args["obj_functions"]).replace(")","").replace("'","").replace("(","").replace("[","").replace("]","")
	#obj = obj.split(',')
	obj = ['spread','seed','communities','fairness','budget','time']
	columns = []
	c = ['min','max','1q','3q','median', 'mean']
	s = []
	for ob in obj:
		for item in c:
			columns.append(f'{ob}-{item}')
	for a in args["population_trend"]:
		t = []
		for j in a:
			for k in j:
				t.append(k)
		s.append(t)
	

	s2 = []
	for a in args["archiver_trend"]:
		t = []
		for j in a:
			for k in j:
				t.append(k)
		s2.append(t)


	df = pd.DataFrame(s2)
	df.to_csv(args["population_file"] + '-population.csv', index=False, header=columns)
	df = pd.DataFrame(s2)
	df.to_csv(args["population_file"] + '-archiver.csv', index=False, header=columns)

	df = pd.DataFrame(args["hypervolume_trend"])
	df.to_csv(args["population_file"] + '-hypervolume.csv', index=False, header=['hv','seed-spread', 'seed-spread-communities', 'seed-spread-fairness', 'seed-spread-budget', 'seed-spread-time'])

	return 


def hypervolume_observer(population, num_generations, num_evaluations, args):
    """
	Updating the Hypervolume list troughout the evolutionaty process
	""" 	
	
	# Switch all the obj. functions' value to -(minus) in order to have a minimization problem and 
	# computed the Hypervolume correctly respect to the pymoo implementation taken by DEAP.
    arch = [list(x.fitness) for x in args["_ec"].population] 
    t = []
    values = []
    arch = [sorted(list(set(x.candidate)), reverse=True) for x in args["_ec"].archive]
    tag = []
    for item in arch:
        if str(item) in args["objective_trend"].keys():
            values.append(args["objective_trend"][str(item)])
            tag.append(item)
        else:
            print('item not found',item)

    #print(len(values), len(arch))
    #print(tag)
    #print(arch)
    assert len(values) == len(arch)
    #if len(values) > 0:
    #    print(values)
	#extract from list values a new list of list only with elements in position 0 and 1
    spread_seed = [x[:2] for x in values]
    spread_seed_comm = [spread_seed[i] + [x[2]] for i,x in enumerate(values)]
    spread_seed_fairness = [spread_seed[i] + [x[3]] for i,x in enumerate(values)]
    spread_seed_budget = [spread_seed[i] + [x[4]] for i,x in enumerate(values)]
    spread_seed_time= [spread_seed[i] + [x[5]] for i,x in enumerate(values)]


    obj = pd.DataFrame(values)

    t = []
    for col in obj.columns:
        l = (obj[col].to_list()) 
        t.append([min(l), max(l), np.percentile(l, 25) , np.percentile(l, 75) , np.median(l),np.mean(l)])
	
    args["population_trend"].append(t)

    obj = pd.DataFrame(values)
    t = []
    for col in obj.columns:
        l = (obj[col].to_list())
        t.append([min(l), max(l), np.percentile(l, 25) , np.percentile(l, 75) , np.median(l),np.mean(l)])
	
    args["archiver_trend"].append(t)

    hv_seed_spread = hypervolume_seed_spread(spread_seed, args)
    hv_seed_spread_comm = hypervolume_with_one(spread_seed_comm, args)
    hv_seed_spread_fairness = hypervolume_with_one(spread_seed_fairness, args)
    hv_seed_spread_budget = hypervolume_with_one(spread_seed_budget, args)
    hv_seed_spread_time = hypervolume_with_one(spread_seed_time, args)
    
    if args["obj_functions"] == ['spread', 'seed']:
        arch = [list(x.fitness) for x in args["_ec"].archive] 
        hv_normal = get_hv(arch, args,no_obj=2)
    elif args["obj_functions"] == ['spread', 'seed', 'communities', 'fairness', 'budget', 'time']:
        arch = [list(x.fitness) for x in args["_ec"].archive] 
        hv_normal = None
    else:
        arch = [list(x.fitness) for x in args["_ec"].archive] 
        hv_normal = get_hv(arch, args, no_obj=3)
	    
    t = [hv_normal, hv_seed_spread, hv_seed_spread_comm, hv_seed_spread_fairness, hv_seed_spread_budget, hv_seed_spread_time]
    #t = []
    args["hypervolume_trend"].append(t)
    #print all hypervolume values in one print with at most 3 digits after the decimal point
	#print all hypervolume values in one print with at most 3 digits after the decimal point
    pr = [hv_seed_spread/hv_seed_spread, hv_seed_spread_comm/hv_seed_spread, hv_seed_spread_fairness/hv_seed_spread, hv_seed_spread_budget/hv_seed_spread, hv_seed_spread_time/hv_seed_spread]
    pr = [round(x,3) for x in pr]
    try:
        print('Generation', num_generations,'hypervolume', round(hv_normal,3), 'others', pr)
    except:
        print('Generation', num_generations,'hypervolume', hv_normal, 'others', pr)
    return