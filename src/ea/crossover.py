import inspyred
from src.ea.mutators import *

@inspyred.ec.variators.crossover
def ea_one_point_crossover(prng, candidate1, candidate2, args):
	#print('Crossing over')
	"""
	Applies 1-point crossover by avoiding repetitions
	"""
	# See common elements.
	common = list(set(candidate1).intersection(set(candidate2)))
	max_trials = 5
	while (len(candidate1) - len(common)) < 2 and max_trials > 0:
		# While candidates have less then 2 nodes not in common crossover would
		# not produce any new candidates, so mutation is forced.
		# E.g., any mutation between (1,2,3,4) and (2,3,4,5) will not produce
		# any new candidate.
		if len(candidate1) - len(common) == 1:
			# If the two candidates differ by 1 element, perform a random mutation
			# once.
			if args["mutation_operator"] == ea_global_random_mutation:
				candidate1 = ea_global_random_mutation(prng, [candidate1], args)[0]
				candidate2 = ea_global_random_mutation(prng, [candidate2], args)[0]
			else:
				candidate1 = ea_local_neighbors_random_mutation(prng, [candidate1], args)[0]
				candidate2 = ea_local_neighbors_random_mutation(prng, [candidate2], args)[0]
		elif len(candidate1) == len(common):
			# If the two candidates are identical, perform a random mutation twice.
			for _ in range(2):
				if args["mutation_operator"] == ea_global_random_mutation:
					candidate1 = ea_global_random_mutation(prng, [candidate1], args)[0]
					candidate2 = ea_global_random_mutation(prng, [candidate2], args)[0]
				else:
					candidate1 = ea_local_neighbors_random_mutation(prng, [candidate1], args)[0]
					candidate2 = ea_local_neighbors_random_mutation(prng, [candidate2], args)[0]

		max_trials -= 1
		common = list(set(candidate1).intersection(set(candidate2)))

	if max_trials==0:
		return [candidate2, candidate1]

	candidate1_to_swap = candidate1.copy()
	candidate2_to_swap = candidate2.copy()
	c1_common = {}
	c2_common = {}

	# get the nodes of each candidate that can be swapped
	for c in common:
		candidate1_to_swap.pop(candidate1_to_swap.index(c))
		candidate2_to_swap.pop(candidate2_to_swap.index(c))
		idx1 = candidate1.index(c)
		idx2 = candidate2.index(c)
		c1_common[idx1] = c
		c2_common[idx2] = c

	# choose swap position

	swap_idx = prng.randint(1, len(candidate1_to_swap) - 1)
	swap = candidate1_to_swap[swap_idx:]
	candidate1_to_swap[swap_idx:] = candidate2_to_swap[swap_idx:]
	candidate2_to_swap[swap_idx:] = swap

	for (idx, c) in c1_common.items():
		candidate1_to_swap.insert(idx, c)
	for (idx, c) in c2_common.items():
		candidate2_to_swap.insert(idx, c)


	return [candidate1_to_swap, candidate2_to_swap]


@inspyred.ec.variators.mutator # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2_super_operator(random, candidate1, args) :
    children = []
    # uniform choice of operator
    randomChoice = random.randint(0,4)
    if len(candidate1) == 1:
        randomChoice = 0
    if randomChoice == 0 :
        children.append(nsga2_insertion_mutation(random, list(candidate1), args) )
    elif randomChoice == 1 :
        children.append(nsga2_removal_mutation(random, list(candidate1), args))
    elif randomChoice == 2:
        children.append(ea_local_neighbors_random_mutation(random, [candidate1], args)[0])
    elif randomChoice == 3:
        children.append(ea_local_neighbors_second_degree_mutation(random, [candidate1], args)[0])
    elif randomChoice == 4:
        children.append(ea_global_low_deg_mutation(random, [candidate1], args)[0])
    
    children = [c for c in children if c is not None and len(c) > 0]
    return children[0]

@inspyred.ec.variators.mutator # decorator that defines the operator as a crossover, even if it isn't in this case :-)
def nsga2_super_operator_base(random, candidate1, args) :
    children = []
    randomChoice = random.randint(0,2)

    if randomChoice == 0 :
        children.append( ea_alteration_mutation(random, list(candidate1), args) )
    elif randomChoice == 1 :
        children.append( nsga2_insertion_mutation(random, list(candidate1), args) )
    elif randomChoice == 2 :
        children.append( nsga2_removal_mutation(random, list(candidate1), args) )

    children = [c for c in children if c is not None and len(c) > 0]
    

    return children[0]

@inspyred.ec.variators.crossover # decorator that defines the operator as a crossover
def nsga2_crossover(random, candidate1, candidate2, args) : 

    children = []   
    max_seed_nodes = args["max_seed_nodes"]

    parent1 = list(set(candidate1))
    parent2 = list(set(candidate2))
    
    # choose random cut point
    cutPoint1 = random.randint(0, len(parent1)-1)
    cutPoint2 = random.randint(0, len(parent2)-1)
    
    # children start as empty lists
    child1 = []
    child2 = []
    
    # swap stuff
    for i in range(0, cutPoint1) : child1.append( parent1[i] )
    for i in range(0, cutPoint2) : child2.append( parent2[i] )
    
    for i in range(cutPoint1, len(parent2)) : child1.append( parent2[i] )
    for i in range(cutPoint2, len(parent1)) : child2.append( parent1[i] )
    
    # reduce children to minimal form
    child1 = list(set(child1))
    child2 = list(set(child2))
    
    # return the two children
    if len(child1) > 0 and len(child1) <= max_seed_nodes : children.append( child1 )
    if len(child2) > 0 and len(child2) <= max_seed_nodes : children.append( child2 )

    return children

def ea_alteration_mutation(random, candidate, args) :
    
    nodes = args["nodes"]
    max_seed_nodes = args["max_seed_nodes"]

    mutatedIndividual = list(set(candidate))
    
    if len(mutatedIndividual) < max_seed_nodes :
        gene = random.randint(0, len(mutatedIndividual)-1)
        mutatedIndividual[gene] = nodes[ random.randint(0, len(nodes)-1) ]
        return mutatedIndividual
    else:
        return candidate

def nsga2_insertion_mutation(random, candidate, args) :
    
    max_seed_nodes = args["max_seed_nodes"]
    nodes = args["nodes"]
    mutatedIndividual = list(set(candidate))

    if len(mutatedIndividual) < max_seed_nodes :
        mutatedIndividual.append( nodes[ random.randint(0, len(nodes)-1) ] )
        return mutatedIndividual
    else:
        return candidate


def nsga2_removal_mutation(random, candidate, args) :
    
    mutatedIndividual = list(set(candidate))
    if len(candidate) > 1 :
        gene = random.randint(0, len(mutatedIndividual)-1)
        mutatedIndividual.pop(gene)
        return mutatedIndividual
    else:
        return candidate