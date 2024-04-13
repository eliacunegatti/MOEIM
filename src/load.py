import networkx as nx


""" Graph loading """

def read_graph(filename, communities, nodetype=int):
	if '_un' in filename:
		G = nx.read_edgelist(filename, create_using=nx.Graph(), nodetype=int)
	elif '_di' in filename:
		G = nx.read_edgelist(filename, create_using=nx.DiGraph(), nodetype=int)

	return G

def read_graph_total(filename, nodetype=int):
	if '_un' in filename:
		graph_class = nx.Graph() # all graph files are undirected
	elif '_di' in filename:
		graph_class = nx.DiGraph() # all graph files are undirected

	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)
	
	return G

