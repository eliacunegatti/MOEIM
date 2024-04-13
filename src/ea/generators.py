

def generator(random, args):
	"""
	simple random generator: generates individual by sampling random nodes
	:param random:
	:param args:
	:return:
	"""
	return random.sample(args["nodes"], args["max_seed_nodes"])


def generator_new_nodes(random, args, new_nodes_percentage=0.9):
	"""
	generator which tries to generate an individual having a specified percentage of nodes not already present
	in the population
	:param random:
	:param args:
	:param new_nodes_percentage:
	:return:
	"""
	new_nodes = int(args["k"]*new_nodes_percentage)
	nodes = args["nodes"].copy()
	population = args["_ec"].population
	pop_nodes = []

	# collect population nodes
	for individual in population:
		for n in individual.candidate:
			if n not in pop_nodes:
				pop_nodes.append(n)

	# separate new nodes
	for node in pop_nodes:
		if node in nodes:
			nodes.remove(node)

	if len(nodes) < new_nodes:
		first_part = nodes
		second_part = random.sample(args["nodes"], args["k"] - len(nodes))
		new_ind = first_part + second_part
	else:
		# new nodes
		first_part = random.sample(nodes, new_nodes)
		# population nodes
		second_part = random.sample(pop_nodes, args["k"]-new_nodes)
		new_ind = first_part + second_part

	# shuffle new and already present nodes
	random.shuffle(new_ind)

	return new_ind


def subpopulation_generator(random, args):
	"""
	for each dimension selects node from one cell
	:param random:
	:param args:
	:return:
	"""
	individual = []
	voronoi_cells = args["voronoi_cells"]
	for i in range(args["k"]):
		nodes = voronoi_cells[list(voronoi_cells.keys())[i]]
		individual.append(random.sample(nodes, 1)[0])

	return individual