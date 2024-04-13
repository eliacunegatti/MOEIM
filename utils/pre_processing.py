
import networkx as nx
import os
import leidenalg
import igraph as ig
import numpy as np
import pandas as pd

def from_networkx_to_igraph(G):
    R = ig.Graph(directed=False)
    R.add_vertices(G.nodes())
    R.add_edges(G.edges())
    return R
def leiden_algorithm(G):
    communities = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    communities = list(communities)
    return communities

def save_groud_truth_communities(communities_threshold, name):
    node_original = []
    comm_original = []
    for idx, item in enumerate(communities_threshold, start=1):
        for node in item:
            node_original.append(node)
            comm_original.append(idx)
    df = pd.DataFrame({'node': node_original, 'comm': comm_original})
    df.to_csv('graph_communities/'+name.replace(".txt","")+'.csv',index=False, sep=",")



r = ''
path = 'graphs/'
for filename in os.listdir(path):
    print(filename)
    how = None
    if '_un' in filename:
        G = nx.read_edgelist(f'{path}{filename}', create_using=nx.Graph())
        print(G.number_of_nodes(), G.number_of_edges())
        Gc = max(nx.connected_components(G), key=len)
        how = 'undirected'
    elif '_di' in filename:
        G = nx.read_edgelist(f'{path}{filename}', create_using=nx.DiGraph())
        print(G.number_of_nodes(), G.number_of_edges())

        strongly_connected_components = list(nx.weakly_connected_components(G))
        Gc = max(strongly_connected_components, key=len)
        #Gc = max(nx.connected_components(G), key=len)
        how = 'directed'
    
    
    G = G.subgraph(Gc)
    #rename nodes to start from 0
    mapping = dict(zip(G, range(0, len(G))))
    G = nx.relabel_nodes(G, mapping)

    print(G.number_of_nodes(), G.number_of_edges())
    R = from_networkx_to_igraph(G)
    print('Saving ....')
    print('Type of graph', how)
    print('Number of nodes: {0}'.format(len(G.nodes())))
    print('Number of edges: {0}'.format(len(G.edges())))

    print('Computing Communities .... ')

    communities = leiden_algorithm(R)
    print('Number of communities detected: {0}'.format(len(communities)))
    
    # delete communities with number of elements < scaling factor
    nodes_to_delete = []
    communities_threshold = []
    for item in communities:
        if len(item) >= 10:
            communities_threshold.append(item)
        else:
            #print('Community {0} deleted'.format(item))
            nodes_to_delete.extend(item)
    
    print('nodes to delete: {0}'.format(len(nodes_to_delete)))

    print('Saving ....')
    print('Type of graph', how)
    print('Number of nodes: {0}'.format(len(G.nodes())))
    print('Number of edges: {0}'.format(len(G.edges())))
    

    r += f'{filename} & {len(G.nodes())} & {len(G.edges())} & {len(communities_threshold)} & {min([len(item) for item in communities_threshold])}& {max([len(item) for item in communities_threshold])}& {round(sum([G.degree(node) for node in G.nodes()])/len(G.nodes()),2)}& {round(np.std([G.degree(node) for node in G.nodes()]),2)}& {max([G.degree(node) for node in G.nodes()])} \\ \\  \n'


    file_to_save = filename
    file_path = f"graphs_cleaned/{file_to_save}"
    save_groud_truth_communities(communities_threshold, file_to_save)

    print('Number of nodes: {0}'.format(len(G.nodes())))
    print('Number of edges: {0}'.format(len(G.edges())))

    nx.write_edgelist(G, file_path)

    print('#'*50)

